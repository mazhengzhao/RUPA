from pathlib import Path
from pypdf import PdfReader, PdfWriter


def split_pdf(input_pdf, output_dir, pages_per_file):
    """
    按指定页数拆分 PDF

    Args:
        input_pdf (str): 输入 PDF 路径
        output_dir (str): 输出目录
        pages_per_file (int): 每个 PDF 包含的页数
    """

    input_pdf = Path(input_pdf)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    reader = PdfReader(str(input_pdf))
    total_pages = len(reader.pages)

    print(f"Input PDF : {input_pdf}")
    print(f"Total pages: {total_pages}")
    print(f"Pages/file: {pages_per_file}")
    print()

    part = 1

    for start in range(0, total_pages, pages_per_file):

        writer = PdfWriter()

        end = min(start + pages_per_file, total_pages)

        for page_idx in range(start, end):
            writer.add_page(reader.pages[page_idx])

        output_name = (
            f"part_{part:03d}_pages_{start+1}-{end}.pdf"
        )

        output_path = output_dir / output_name

        with open(output_path, "wb") as f:
            writer.write(f)

        print(f"Saved: {output_name}")

        part += 1

    print("\nDone!")


if __name__ == "__main__":

    input_pdf = "_AAAI_2027__Trajectory_Awareness_Uncertainty_Propogation (8).pdf"      # 修改为你的 PDF
    output_dir = "output"        # 输出目录
    pages_per_file = 9         # 每个 PDF 页数

    split_pdf(input_pdf, output_dir, pages_per_file)