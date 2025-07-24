import os
import pdb as debugger

os.environ["TESSDATA_PREFIX"] = r"C:\Program Files\Tesseract-OCR\tessdata"

pdb = debugger.set_trace

if __name__ == '__main__':
    import fitz

    doc = fitz.open(r"C:\Users\fatih\Downloads\enonce-examen-2019.pdf")
    doc2 = fitz.open(r"C:\Users\fatih\Downloads\palum_couts_2015.pdf")

    page = doc2[1]
    tp = page.get_textpage_ocr(language='fra', full=True, dpi=120)
    # dpi = 120 gives good results
    # (maybe add a post OCR language check using AI)
    text = page.get_text(textpage=tp)
    lines = text.split("\n")
    for line in lines:
        print(line)

    pdb()
