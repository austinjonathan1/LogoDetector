from pdf2image import convert_from_path, convert_from_bytes
from pdf2image.exceptions import (PDFInfoNotInstalledError, PDFPageCountError, PDFSyntaxError)
from absl import app, flags, logging
from absl.flags import FLAGS

flags.DEFINE_integer('count', 1, 'number of image/excel sets')

# WEIGHTS, CLASSES, & FLAGS ROUTING
def main(_argv):
    if FLAGS.count:
        count = FLAGS.count
    excel = []
    images = []
    for i in range(count):
        con = convert_from_path('data/pdf/test (' + str(i+1) + ').pdf', output_folder='data/pdf', fmt="jpg", single_file=True, output_file='test (' + str(i+1) + ')')
        excel.append('data/excel/test (' + str(i+1) + ').xlsx')
        images.append('data/images/test (' + str(i+1) + ').jpg')

# MAIN
if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass

#EX: images = convert_from_path('data/pdf/pair.pdf', output_folder='data/pdf', fmt="jpg", single_file=True, output_file='test')
