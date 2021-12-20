import os
import glob
from zipfile import ZipFile


def create_submission_zip(rootdir=None, zip_path='cs484submission.zip'):
    if rootdir is None:
        os.path.dirname(os.path.abspath(__file__))

    code_to_include = glob.glob(os.path.join('code', '*.py'))
    # No questions in hw3
    # tex_to_compile = glob.glob(os.path.join('questions', '20??_*_Homework?_Questions.tex'))
    tex_to_compile = glob.glob(os.path.join('writeup', '*.tex'))

    with ZipFile(zip_path, mode='w') as zf:
        # Include all code files 'code/*.py'.
        for code in code_to_include:
            zf.write(code)

        # Include questions and writeup pdfs after tex compilation if needed.
        for tex in tex_to_compile:
            pdf = os.path.splitext(tex)[0] + '.pdf'
            if not os.path.exists(pdf):
                os.system(' && '.join([
                    f'cd {os.path.dirname(tex)}',
                    f'pdflatex {os.path.basename(tex)}',
                    f'pdflatex {os.path.basename(tex)}',
                ]))
            zf.write(pdf)

    print('========================================================')
    print('Created:', zip_path)
    print('Before submission, rename it to hw?_student-id_name.zip')


if __name__ == '__main__':
    create_submission_zip()
