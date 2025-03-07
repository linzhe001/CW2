import PyPDF2
import sys

def pdf_to_txt(pdf_path, txt_path):
    """将PDF文件转换为TXT文件"""
    try:
        # 打开PDF文件
        with open(pdf_path, 'rb') as pdf_file:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            # 打开/创建TXT文件
            with open(txt_path, 'w', encoding='utf-8') as txt_file:
                # 逐页读取并提取文本
                for page_num, page in enumerate(pdf_reader.pages):
                    text = page.extract_text()
                    if text:
                        txt_file.write(text)
                        txt_file.write("\n")
        print(f"转换成功，TXT文件已保存为：{txt_path}")
    except Exception as e:
        print("转换过程中出现错误：", e)

if __name__ == '__main__':
    if len(sys.argv) == 1:
        # 未提供任何参数时，使用默认的PDF文件路径
        print("未检测到命令行参数，使用默认文件：./coursework_2.pdf")
        pdf_to_txt('./coursework_2.pdf', 'coursework_2.txt')
    elif len(sys.argv) == 3:
        input_pdf = sys.argv[1]
        output_txt = sys.argv[2]
        pdf_to_txt(input_pdf, output_txt)
    else:
        print("用法: python pdf2txt.py 输入.pdf 输出.txt")