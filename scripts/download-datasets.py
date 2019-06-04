import optparse
import os

parser = optparse.OptionParser()
parser.add_option('--out', type='sting', default='/home/fr/datasets')
(options, args) = parser.parse_args()
out = options.out

http_user = 'zhangdo@uestc.edu.cn'
http_password = '1O|)8f+%Z8'


def main():
    wget_command = f'wget http://megaface.cs.washington.edu/dataset/download/content/MegaFace_dataset.tar.gz -P {out} '\
                   f'--http-user={http_user} --http-password={http_password}'
    os.system(wget_command)


if __name__ == '__main__':
    main()
