import time
import uuid
from copy import deepcopy
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Optional

from loguru import logger

from tau2.agent.base import BaseAgent, is_valid_agent_history_message
from tau2.agent.llm_agent import LLMSoloAgent
from tau2.data_model.message import (
    AssistantMessage,
    Message,
    MultiToolMessage,
    ToolMessage,
    UserMessage,
)
from tau2.data_model.simulation import SimulationRun, TerminationReason
from tau2.data_model.tasks import EnvFunctionCall, InitializationData, Task
from tau2.environment.environment import Environment, EnvironmentInfo
from tau2.user.base import BaseUser, is_valid_user_history_message
from tau2.user.user_simulator import DummyUser, UserSimulator, UserState
from tau2.metrics.uncertainty import (
    EmbeddingService,
    TRACERConfig,
    calculate_inference_gap,
    calculate_hybrid_repetition_score,
    calculate_tool_repetition,
    calculate_tracer_from_trajectory,
    get_uncertainty_stats,
)
from tau2.utils.llm_utils import get_cost
from tau2.utils.utils import format_time, get_now


class Role(str, Enum):
    AGENT = "agent"
    USER = "user"
    ENV = "env"


DEFAULT_FIRST_AGENT_MESSAGE = AssistantMessage(
    role="assistant", content="Hi! How can I help you today?", cost=0.0
)


class Orchestrator:
    """
    Orchestrator for the simulation given a task.
    Passes messages between the Agent, User, and Environment.
    """

    def __init__(
        self,
        domain: str,
        agent: BaseAgent,
        user: BaseUser,
        environment: Environment,
        task: Task,
        max_steps: int = 100,
        max_errors: int = 10,
        seed: Optional[int] = None,
        solo_mode: bool = False,
        calculate_uncertainty: bool = False,
        tracer_config: Optional[TRACERConfig] = None,
    ):
        self.domain = domain
        self.agent = agent
        self.user = user
        self.environment = environment
        self.task = task
        self.seed = seed
        self.solo_mode = solo_mode
        self.calculate_uncertainty = calculate_uncertainty
        self.agent_state: Optional[Any] = None
        self.user_state: Optional[UserState] = None
        self.trajectory: list[Message] = []
        self.max_steps = max_steps
        self.max_errors = max_errors
        self.step_count = 0
        self.done = False
        self.termination_reason: Optional[TerminationReason] = None
        self.num_errors = 0
        self.from_role: Optional[Role] = None
        self.to_role: Optional[Role] = None
        self.message: Optional[Message] = None
        
        # TRACER configuration
        self.tracer_config = tracer_config if tracer_config is not None else TRACERConfig()
        
        # Situational awareness tracking (reuses calculate_uncertainty flag)
        self.calculate_situational_awareness = calculate_uncertainty
        self.conversation_history: list[str] = []
        self.agent_history: list[str] = []  # Track agent text responses for repetition detection
        self.agent_tool_history: list[list] = []  # Track agent tool calls for duplicate detection
        self.initial_instruction: Optional[str] = None
        self.last_agent_message: Optional[str] = None
        
        # Initialize embedding service if enabled
        if self.calculate_situational_awareness:
            self.embedding_service = EmbeddingService()
            # Extract initial user instruction from task
            if task.user_scenario and hasattr(task.user_scenario, 'instructions'):
                instructions = task.user_scenario.instructions
                if isinstance(instructions, dict):
                    # Convert dict to string representation
                    self.initial_instruction = str(instructions)
                else:
                    self.initial_instruction = str(instructions)
            logger.debug(f"Situational awareness enabled. Initial instruction: {self.initial_instruction[:100] if self.initial_instruction else 'None'}...")
        else:
            self.embedding_service = None

    def initialize(self):
        """
        Initialize the orchestrator.
        - If the tasks specifies an initial state, use it to initialize the environment.
        - Initialize the agent and user states.
        - Send the first message (default message from the agent to the user).
        """
        initial_state = self.task.initial_state
        initialization_data = (
            initial_state.initialization_data if initial_state is not None else None
        )
        initialization_actions = (
            initial_state.initialization_actions if initial_state is not None else None
        )
        message_history = (
            deepcopy(initial_state.message_history)
            if initial_state is not None and initial_state.message_history is not None
            else []
        )
        for msg in message_history:
            msg.turn_idx = None

        # Add timestamps to the message history
        message_history = self._add_timestamps(message_history)

        if self.solo_mode:
            assert self.environment.solo_mode, "Environment should be in solo mode"
            assert isinstance(self.agent, LLMSoloAgent), (
                "Agent must be a LLMSoloAgent in solo mode"
            )
            assert isinstance(self.user, DummyUser), (
                "User must be a DummyUser in solo mode"
            )

        # Initialize Environment state
        self._initialize_environment(
            initialization_data=initialization_data,
            initialization_actions=initialization_actions,
            message_history=message_history,
        )

        # Set seeds for the agent, user
        if self.seed is not None:
            self.agent.set_seed(self.seed)
            self.user.set_seed(self.seed)

        # Initialize the agent and user states
        if len(message_history) > 0:
            self.validate_message_history(message_history)

            last_message = message_history[-1]
            # Last message is an assistant message
            if isinstance(last_message, AssistantMessage):
                self.from_role = Role.AGENT
                if not last_message.is_tool_call():  # Last message is for the user
                    self.to_role = Role.USER
                else:  # Last message is for the environment
                    self.to_role = Role.ENV
                self.agent_state = self.agent.get_init_state(
                    message_history=[
                        msg
                        for msg in message_history
                        if is_valid_agent_history_message(msg)
                    ]
                )
                self.user_state = self.user.get_init_state(
                    message_history=[
                        msg
                        for msg in message_history[:-1]
                        if is_valid_user_history_message(msg)
                    ]
                )
                self.message = last_message
                if self.agent.is_stop(last_message):
                    self.done = True
                    self.termination_reason = TerminationReason.AGENT_STOP
            # Last message is a user message
            elif isinstance(last_message, UserMessage):
                self.from_role = Role.USER
                if not last_message.is_tool_call():  # Last message is for the agent
                    self.to_role = Role.AGENT
                else:  # Last message is for the environment
                    self.to_role = Role.ENV
                self.user_state = self.user.get_init_state(
                    message_history=[
                        msg
                        for msg in message_history
                        if is_valid_user_history_message(msg)
                    ]
                )
                self.agent_state = self.agent.get_init_state(
                    message_history=[
                        msg
                        for msg in message_history[:-1]
                        if is_valid_agent_history_message(msg)
                    ]
                )
                self.message = last_message
                self.done = UserSimulator.is_stop(last_message)
                if self.done:
                    self.termination_reason = TerminationReason.USER_STOP
            # Last message is a tool message
            elif isinstance(last_message, ToolMessage):
                self.from_role = Role.ENV
                if last_message.requestor == "assistant":
                    self.to_role = Role.AGENT
                    self.agent_state = self.agent.get_init_state(
                        message_history=[
                            msg
                            for msg in message_history[:-1]
                            if is_valid_agent_history_message(msg)
                        ]
                    )
                    self.user_state = self.user.get_init_state(
                        message_history=[
                            msg
                            for msg in message_history
                            if is_valid_user_history_message(msg)
                        ]
                    )
                else:
                    self.to_role = Role.USER
                    self.agent_state = self.agent.get_init_state(
                        message_history=[
                            msg
                            for msg in message_history
                            if is_valid_agent_history_message(msg)
                        ]
                    )
                    self.user_state = self.user.get_init_state(
                        message_history=[
                            msg
                            for msg in message_history[:-1]
                            if is_valid_user_history_message(msg)
                        ]
                    )
                self.message = last_message
            else:
                raise ValueError(
                    f"Last message should be of type AssistantMessage, UserMessage, or ToolMessage, got {type(last_message)}"
                )
            self.trajectory = message_history

        else:
            self.agent_state = self.agent.get_init_state()
            self.user_state = self.user.get_init_state()
            if not self.solo_mode:
                first_message = deepcopy(DEFAULT_FIRST_AGENT_MESSAGE)
                first_message.timestamp = get_now()
                self.trajectory = [first_message]
                self.message = first_message
                self.from_role = Role.AGENT
                self.to_role = Role.USER
            else:
                first_message, agent_state = self.agent.generate_next_message(
                    None, self.agent_state
                )
                self.trajectory = [first_message]
                self.message = first_message
                self.from_role = Role.AGENT
                self.to_role = Role.ENV
                self.done = self.agent.is_stop(first_message)
                if self.done:
                    self.termination_reason = TerminationReason.AGENT_STOP

        self.environment.sync_tools()

    def run(self) -> SimulationRun:
        """
        Run the simulation.

        Returns:
            SimulationRun: The simulation run.
        """
        start_time = get_now()
        start = time.perf_counter()
        self.initialize()
        while not self.done:
            self.step()
            if self.step_count >= self.max_steps:
                self.done = True
                self.termination_reason = TerminationReason.MAX_STEPS
            if self.num_errors >= self.max_errors:
                self.done = True
                self.termination_reason = TerminationReason.TOO_MANY_ERRORS
        duration = time.perf_counter() - start
        messages = self.get_trajectory()
        res = get_cost(messages)
        if res is None:
            agent_cost, user_cost = None, None
        else:
            agent_cost, user_cost = res
        
        # Calculate TRACER aggregation score if uncertainty is enabled
        tracer_metrics = None
        if self.calculate_uncertainty:
            try:
                tracer_result = calculate_tracer_from_trajectory(messages, self.tracer_config)
                # Remove penalties list from output (too verbose for JSON)
                tracer_metrics = {k: v for k, v in tracer_result.items() if k != 'penalties'}
                logger.info(
                    f"TRACER Score: {tracer_metrics['tracer_score']:.4f} "
                    f"(N={tracer_metrics['num_steps']}, mean_penalty={tracer_metrics['mean_penalty']:.4f})"
                )
            except Exception as e:
                logger.warning(f"Failed to calculate TRACER score: {e}")
                tracer_metrics = None
        
        simulation_run = SimulationRun(
            id=str(uuid.uuid4()),
            task_id=self.task.id,
            start_time=start_time,
            end_time=get_now(),
            duration=duration,
            termination_reason=self.termination_reason.value,
            reward_info=None,
            user_cost=user_cost,
            agent_cost=agent_cost,
            messages=messages,
            seed=self.seed,
            tracer_metrics=tracer_metrics,
        )
        return simulation_run

    def _format_message_for_history(self, message: Message) -> str:
        """
        Format a message for inclusion in conversation history.
        
        Converts messages to readable string format for semantic distance calculation.
        
        Args:
            message: Message to format
            
        Returns:
            str: Formatted message string
        """
        if isinstance(message, (AssistantMessage, UserMessage)):
            parts = []
            
            # Add text content
            if message.content:
                parts.append(message.content)
            
            # Add tool calls description
            if message.tool_calls:
                for tool_call in message.tool_calls:
                    tool_str = f"Tool: {tool_call.name}({tool_call.arguments})"
                    parts.append(tool_str)
            
            return " | ".join(parts) if parts else ""
        
        elif isinstance(message, ToolMessage):
            # Format tool observation
            if message.content:
                return f"Observation: {message.content}"
            return "Observation: [empty]"
        
        return ""
    
    def _elicit_self_assessed_confidence(self, message: Message) -> Optional[float]:
        """
        Elicit self-assessed confidence from the agent or user for their response.
        
        Makes a secondary LLM call to ask the agent/user to rate their confidence
        in the response they just provided. This is used as a baseline metric
        for comparison with TRACER.
        
        Args:
            message: The AssistantMessage or UserMessage to elicit confidence for
            
        Returns:
            Optional[float]: Confidence score in [0.0, 1.0], or None if elicitation fails
        """
        try:
            # Determine if this is an agent or user message
            is_agent = isinstance(message, AssistantMessage)
            is_user = isinstance(message, UserMessage)
            
            if not (is_agent or is_user):
                logger.debug("Message is neither AssistantMessage nor UserMessage")
                return None
            
            # Format the response for the confidence prompt
            response_parts = []
            if message.content:
                response_parts.append(message.content)
            if hasattr(message, 'tool_calls') and message.tool_calls:
                for tool_call in message.tool_calls:
                    tool_str = f"Tool call: {tool_call.name}({tool_call.arguments})"
                    response_parts.append(tool_str)
            
            if not response_parts:
                logger.debug("No content to elicit confidence for")
                return None
            
            response_text = " | ".join(response_parts)
            
            # Construct the confidence elicitation prompt (different for agent vs user)
            if is_agent:
                confidence_prompt = f"""You just provided the following response:
{response_text}

TASK: Rate your confidence that this response is correct, helpful, and will successfully address the user's need.

IMPORTANT: Be honest and precise. Overconfidence is harmful. Consider:
- Do you have all necessary information to answer correctly?
- Are there any ambiguities or assumptions in your response?
- Could there be alternative interpretations or solutions?
- Are you making any tool calls without being certain of the parameters?
- Is the user's request clear, or are you inferring their intent?

Report your confidence as a single number from 0.0 to 1.0:
- 1.0 = Completely certain: You have all information and the response is definitively correct
- 0.8 = High confidence: Very likely correct, minor uncertainty about edge cases
- 0.6 = Moderate confidence: Probably correct, but noticeable uncertainty or assumptions
- 0.4 = Low confidence: Significant uncertainty, multiple possible interpretations
- 0.2 = Very uncertain: Guessing or lacking critical information
- 0.0 = No confidence: Cannot determine if response is appropriate

Be calibrated: If you're unsure, report lower confidence. Accuracy matters more than optimism.

Your confidence score (just the number):"""
            else:  # is_user
                confidence_prompt = f"""You just provided the following message:
{response_text}

TASK: Rate your confidence that this message clearly and correctly expresses your need/request.

IMPORTANT: Be honest and precise. Consider:
- Is your request clear and unambiguous?
- Do you have all the information needed to make this request?
- Are you certain about what you're asking for?
- Could the agent misinterpret your request?
- Are you making assumptions about what's available or possible?

Report your confidence as a single number from 0.0 to 1.0:
- 1.0 = Completely certain: Your request is crystal clear and you know exactly what you need
- 0.8 = High confidence: Request is clear, minor ambiguity possible
- 0.6 = Moderate confidence: Somewhat clear, but some uncertainty about specifics
- 0.4 = Low confidence: Significant uncertainty about what you're requesting
- 0.2 = Very uncertain: Unclear about your needs or how to express them
- 0.0 = No confidence: Very confused or unable to articulate your request

Be calibrated: If you're unsure, report lower confidence. Honesty matters more than optimism.

Your confidence score (just the number):"""
            
            # Prepare LLM arguments (copy and override specific settings)
            from copy import deepcopy
            from tau2.data_model.message import UserMessage as ConfidenceUserMessage
            from tau2.utils.llm_utils import generate
            
            # Get the appropriate LLM and args based on message type
            if is_agent:
                llm = self.agent.llm
                llm_args_base = self.agent.llm_args
            else:  # is_user
                llm = self.user.llm
                llm_args_base = self.user.llm_args
            
            llm_args_copy = deepcopy(llm_args_base)
            llm_args_copy.update({
                "temperature": 0.1,
                "max_tokens": 500,  # Increased to handle longer prompts and prevent truncation
                "logprobs": False,  # Don't need logprobs for confidence query
            })
            
            # Make the secondary LLM call
            confidence_message = generate(
                model=llm,
                messages=[ConfidenceUserMessage(role="user", content=confidence_prompt)],
                tools=None,  # No tools needed for this query
                **llm_args_copy
            )
            
            # Parse the response
            if not confidence_message.content:
                logger.debug("Empty response from confidence elicitation (this is non-critical)")
                return None
            
            # Extract float value from response
            import re
            confidence_text = confidence_message.content.strip()
            
            # Try to extract a number (handles various formats)
            number_match = re.search(r'(\d+\.?\d*)', confidence_text)
            if number_match:
                confidence_value = float(number_match.group(1))
                
                # Clamp to [0.0, 1.0]
                confidence_value = max(0.0, min(1.0, confidence_value))
                
                logger.debug(f"Elicited self-assessed confidence: {confidence_value:.4f}")
                return confidence_value
            else:
                logger.debug(f"Could not parse confidence value from response: {confidence_text[:50]}...")
                return None
                
        except Exception as e:
            logger.debug(f"Confidence elicitation failed (non-critical): {e}")
            return None
    
    def _calculate_and_attach_metrics(
        self, message: Message
    ) -> None:
        """
        Calculate and attach uncertainty and semantic distance metrics to a message.
        
        Calculates:
        - Uncertainty statistics (U_i): normalized entropy from logprobs
        - Hybrid Repetition (Da): combines semantic + lexical similarity for agent messages
        - Tool Repetition: exact duplicate tool call detection
        - Inference Gap (Do): semantic distance between action and observation
        
        Args:
            message: The message to calculate metrics for
        """
        if not self.calculate_uncertainty:
            return
        
        # Only calculate metrics for agent and user messages
        if not isinstance(message, (AssistantMessage, UserMessage)):
            return
        
        # Calculate uncertainty statistics from logprobs
        if hasattr(message, 'logprobs') and message.logprobs is not None:
            uncertainty_stats = get_uncertainty_stats(message.logprobs)
            # Store as dictionary for JSON serialization
            message.uncertainty = uncertainty_stats.model_dump()
            logger.debug(
                f"Calculated uncertainty for {message.role} message: "
                f"U_i={uncertainty_stats.normalized_entropy:.4f} "
                f"(tokens={uncertainty_stats.token_count})"
            )
        
        # Elicit self-assessed confidence for AGENT and USER messages
        # This is a baseline metric for comparison with TRACER
        if isinstance(message, (AssistantMessage, UserMessage)):
            confidence = self._elicit_self_assessed_confidence(message)
            if message.uncertainty is None:
                message.uncertainty = {}
            message.uncertainty['self_assessed_confidence'] = confidence
            if confidence is not None:
                actor_type = "agent" if isinstance(message, AssistantMessage) else "user"
                logger.debug(f"Self-assessed confidence ({actor_type}): {confidence:.4f}")
        
        # Calculate semantic distance metrics if situational awareness enabled
        if self.calculate_situational_awareness and self.embedding_service:
            # Calculate Da (Hybrid Repetition) for AGENT messages only
            # Combines semantic similarity with lexical overlap to distinguish looping from enumeration
            if isinstance(message, AssistantMessage):
                try:
                    # Get current message content
                    current_text = self._format_message_for_history(message)
                    
                    # Calculate text-based hybrid repetition
                    text_repetition = 0.0
                    if current_text:
                        text_repetition = calculate_hybrid_repetition_score(
                            current_text,
                            self.agent_history
                        )
                        # Update agent history for next iteration
                        self.agent_history.append(current_text)
                    
                    # Calculate tool-based repetition
                    tool_repetition = 0.0
                    if message.tool_calls:
                        tool_repetition = calculate_tool_repetition(
                            message.tool_calls,
                            self.agent_tool_history
                        )
                        # Update tool history
                        self.agent_tool_history.append(message.tool_calls)
                    
                    # Aggregate: max (worst-case penalty)
                    # Either text looping OR tool duplication is a failure signal
                    da_score = max(text_repetition, tool_repetition)
                    message.da_score = da_score
                    
                    logger.debug(
                        f"Calculated Da for agent: text_rep={text_repetition:.4f}, "
                        f"tool_rep={tool_repetition:.4f}, final={da_score:.4f}"
                    )
                except Exception as e:
                    logger.warning(f"Failed to calculate repetition score: {e}")
                    message.da_score = None

    def step(self):
        """
        Perform one step of the simulation.
        Sends self.message from self.from_role to self.to_role
        This can either be a message from agent to user/environment, environment to agent, or user to agent
        Updates self.trajectory
        """
        if self.done:
            raise ValueError("Simulation is done")
        logger.debug(
            f"Step {self.step_count}. Sending message from {self.from_role} to {self.to_role}"
        )
        logger.debug(
            f"Step {self.step_count}.\nFrom role: {self.from_role}\nTo role: {self.to_role}\nMessage: {self.message}"
        )
        # AGENT/ENV -> USER
        if self.from_role in [Role.AGENT, Role.ENV] and self.to_role == Role.USER:
            user_msg, self.user_state = self.user.generate_next_message(
                self.message, self.user_state
            )
            user_msg.validate()
            
            # Add to conversation history
            if self.calculate_situational_awareness:
                msg_text = self._format_message_for_history(user_msg)
                if msg_text:
                    self.conversation_history.append(msg_text)
            
            # Calculate metrics (uncertainty + semantic distance)
            self._calculate_and_attach_metrics(user_msg)
            
            # Calculate Do (Inference Gap) for User Coherence
            # Antecedent = agent's last message, Consequent = user's response
            if self.calculate_situational_awareness and self.last_agent_message:
                try:
                    user_response = self._format_message_for_history(user_msg)
                    if user_response:
                        do_score = calculate_inference_gap(
                            self.last_agent_message,
                            user_response
                        )
                        user_msg.do_score = do_score
                        user_msg.do_type = "user_coherence"
                        logger.debug(f"Calculated Do (user coherence): {do_score:.4f}")
                except Exception as e:
                    logger.warning(f"Failed to calculate Do for user: {e}")
            
            if UserSimulator.is_stop(user_msg):
                self.done = True
                self.termination_reason = TerminationReason.USER_STOP
            self.trajectory.append(user_msg)
            self.message = user_msg
            self.from_role = Role.USER
            if user_msg.is_tool_call():
                self.to_role = Role.ENV
            else:
                self.to_role = Role.AGENT
        # USER/ENV -> AGENT
        elif (
            self.from_role == Role.USER or self.from_role == Role.ENV
        ) and self.to_role == Role.AGENT:
            agent_msg, self.agent_state = self.agent.generate_next_message(
                self.message, self.agent_state
            )
            agent_msg.validate()
            
            # Add to conversation history
            if self.calculate_situational_awareness:
                msg_text = self._format_message_for_history(agent_msg)
                if msg_text:
                    self.conversation_history.append(msg_text)
                    # Store last agent message for user coherence calculation
                    self.last_agent_message = msg_text
            
            # Calculate metrics (uncertainty + semantic distance)
            self._calculate_and_attach_metrics(agent_msg)
            
            if self.agent.is_stop(agent_msg):
                self.done = True
                self.termination_reason = TerminationReason.AGENT_STOP
            self.trajectory.append(agent_msg)
            self.message = agent_msg
            self.from_role = Role.AGENT
            if agent_msg.is_tool_call():
                self.to_role = Role.ENV
            else:
                self.to_role = Role.USER
        # AGENT/USER -> ENV
        elif self.from_role in [Role.AGENT, Role.USER] and self.to_role == Role.ENV:
            if not self.message.is_tool_call():
                raise ValueError("Agent or User should send tool call to environment")
            
            # Store the tool call message for Do calculation
            tool_call_msg = self.message
            tool_call_text = self._format_message_for_history(tool_call_msg) if self.calculate_situational_awareness else None
            
            tool_msgs = []
            for tool_call in self.message.tool_calls:
                tool_msg = self.environment.get_response(tool_call)
                tool_msgs.append(tool_msg)
            assert len(self.message.tool_calls) == len(tool_msgs), (
                "Number of tool calls and tool messages should be the same"
            )
            
            # Calculate Do (Inference Gap) for Agent/User Tool Coherence
            if self.calculate_situational_awareness and tool_call_text:
                try:
                    # Concatenate all tool observations
                    tool_observations = []
                    for tool_msg in tool_msgs:
                        obs = self._format_message_for_history(tool_msg)
                        if obs:
                            tool_observations.append(obs)
                    
                    if tool_observations:
                        combined_observation = " | ".join(tool_observations)
                        do_score = calculate_inference_gap(
                            tool_call_text,
                            combined_observation
                        )
                        
                        # Attach Do score to the requesting message (agent or user)
                        if self.from_role == Role.AGENT:
                            tool_call_msg.do_score = do_score
                            tool_call_msg.do_type = "agent_coherence"
                            logger.debug(f"Calculated Do (agent coherence): {do_score:.4f}")
                        elif self.from_role == Role.USER:
                            tool_call_msg.do_score = do_score
                            tool_call_msg.do_type = "user_coherence"
                            logger.debug(f"Calculated Do (user tool coherence): {do_score:.4f}")
                        
                        # Add tool observation to history
                        self.conversation_history.append(combined_observation)
                except Exception as e:
                    logger.warning(f"Failed to calculate Do for tool call: {e}")
            
            self.trajectory.extend(tool_msgs)
            if (
                len(tool_msgs) > 1
            ):  # Packaging multiple tool messages into a MultiToolMessage
                self.message = MultiToolMessage(
                    role="tool",
                    tool_messages=tool_msgs,
                )
            else:
                self.message = tool_msgs[0]
            self.to_role = self.from_role
            self.from_role = Role.ENV
        else:
            raise ValueError(
                f"Invalid role combination. From role: {self.from_role}, To role: {self.to_role}"
            )
        self.step_count += 1
        self.environment.sync_tools()

    def get_trajectory(self) -> list[Message]:
        """
        Get the trajectory of the simulation.
        The trajectory is sorted by timestamp, turn_idx are added to messages, trajectory is returned.
        """
        messages: list[Message] = sorted(
            deepcopy(self.trajectory),
            key=lambda x: x.timestamp,
        )
        trajectory = []
        for i, msg in enumerate(messages):
            msg = deepcopy(msg)
            msg.turn_idx = i
            trajectory.append(msg)
        return trajectory

    @classmethod
    def validate_message_history(cls, message_history: list[Message]):
        """
        Validate a message history.
            - Should only contain AssistantMessage, UserMessage, ToolMessage
            - All assistant/user messages should be either to user or tool call, not both.
            - If n tool calls are made by a participant, exactly n tool messages should follow with requestor matching the participant.
        """
        num_expected_tool_messages = 0
        requestor = None
        for msg in message_history:
            if isinstance(msg, AssistantMessage) or isinstance(msg, UserMessage):
                msg.validate()
                if msg.is_tool_call():
                    if num_expected_tool_messages > 0:
                        raise ValueError(
                            f"{num_expected_tool_messages} tool messages are missing. Got {msg.role} message."
                        )
                    num_expected_tool_messages = len(msg.tool_calls)
                    requestor = msg.role
                else:
                    num_expected_tool_messages == 0
                    requestor = None
            elif isinstance(msg, ToolMessage):
                if num_expected_tool_messages == 0 or requestor is None:
                    raise ValueError("No tool messages expected.")
                if requestor != msg.requestor:
                    raise ValueError(
                        f"Got tool message from {msg.requestor}, expected {requestor}."
                    )
                num_expected_tool_messages -= 1
            else:
                raise ValueError(f"Invalid message type: {type(msg)}")

    def _initialize_environment(
        self,
        initialization_data: Optional[InitializationData],
        initialization_actions: Optional[list[EnvFunctionCall]],
        message_history: list[Message],
    ):
        """
        Initialize the environment.
        """
        self.environment.set_state(
            initialization_data=initialization_data,
            initialization_actions=initialization_actions,
            message_history=message_history,
        )

    def _get_environment_info(self) -> EnvironmentInfo:
        """
        Get the environment info.
        """
        return self.environment.get_info()

    def _count_errors(self, message_history: list[Message]) -> int:
        """
        Count the number of errors in the message history.
        """
        return sum(
            1 for msg in message_history if isinstance(msg, ToolMessage) and msg.error
        )

    def _add_timestamps(
        self, message_history: list[Message]
    ) -> list[tuple[str, Message]]:
        """
        Add timestamps to the message history.
        This is used to sort the messages by timestamp.
        """
        time_offset = datetime.now() - timedelta(seconds=len(message_history))
        for i, msg in enumerate(message_history):
            msg.timestamp = format_time(time_offset + timedelta(seconds=i))
        return message_history
