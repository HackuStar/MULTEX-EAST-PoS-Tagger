"""
Downloader for multex east corpus.
"""

import os
from os.path import expanduser, abspath
import sys
import urllib
import zipfile

import nltk.data

isCustomPath = False


def main():
    download()


def download():
    try:
        __download__()
    except KeyboardInterrupt:
        print("\nDiscarded download due to keyboard interrupt.\n")


def __getFilePath__():
    global isCustomPath
    paths = list(zip(range(len(nltk.data.path)+1), nltk.data.path + ["custom"]))

    pathStr = ""
    try:
        pathStr = raw_input("Where should the corpus be saved?" + str(paths) + " [%s]: " % 0)
    except:
        pathStr = input("Where should the corpus be saved?" + str(paths) + " [%s]: " % 0)

    pathNum = None
    if pathStr:
        pathNum = int(pathStr)
    else:
        pathNum = 0

    if (pathNum == len(nltk.data.path)):
        isCustomPath = True
        try:
            return abspath(raw_input(
            "Please input the directory where you want the files to be saved (NO backslash at the end): ")) + "/"
        except:
            return abspath(input(
            "Please input the directory where you want the files to be saved (NO backslash at the end): ")) + "/"
    else:
        return abspath(nltk.data.path[pathNum]) + "/corpora/"


def __download__():
    filePath = __getFilePath__()
    finished = False

    try:
        if not os.path.exists(filePath):
            os.makedirs(filePath)
    except EnvironmentError:
        print("Could not create or write to file")
    else:
        # download zip archive
        with open(filePath + "mte_teip5.zip", "wb") as f:
            url = "https://www.clarin.si/repository/xmlui/bitstream/handle/11356/1043/MTE1984-ana.zip"
            try:
                request = urllib.urlopen(url)
            except:
                request = urllib.request.urlopen(url)
            chunk_read_write(f, request, report_hook=chunk_report)
            print("Download finished")

        # handle "invalid" zip format from clarin.si
        with open(filePath + "mte_teip5.zip", "r+b") as f:
            content = f.read()
            pos = content.rfind(
                b'\x50\x4b\x05\x06')  # reverse find: this string of bytes is the end of the zip's central directory.
            if pos > 0:
                f.seek(pos + 20)  # +20: see secion V.I in 'ZIP format' link above.
                f.truncate()
                f.write(b'\x00\x00')  # Zip file comment length: 0 byte length; tell zip applications to stop reading.
                f.seek(0)

        # extract zip archive
        print("Extracting files...")
        with zipfile.ZipFile(filePath + "mte_teip5.zip", "r") as z:
            z.extractall(filePath)
        os.rename(filePath + "MTE1984-ana", filePath + "mte_teip5")

    print("Done")


def chunk_report(bytes_so_far, chunk_size, total_size):
    percent = float(bytes_so_far) / total_size
    percent = round(percent * 100, 2)
    sys.stdout.write("Downloaded %d of %d bytes (%0.2f%%)\r" %
                     (bytes_so_far, total_size, percent))

    if bytes_so_far >= total_size:
        sys.stdout.write('\n')


def chunk_read_write(fileHandle, response, chunk_size=8192, report_hook=None):
    try:
        total_size = response.info().getheader('Content-Length').strip()
    except:
        total_size = response.getheader('Content-Length').strip()
    total_size = int(total_size)
    bytes_so_far = 0

    while 1:
        chunk = response.read(chunk_size)
        fileHandle.write(chunk)
        bytes_so_far += len(chunk)

        if not chunk:
            break

        if report_hook:
            report_hook(bytes_so_far, chunk_size, total_size)

    return bytes_so_far


if __name__ == "__main__":
    main()
