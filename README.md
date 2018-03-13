This repo is forked from [rorodata/background-removal](https://github.com/rorodata/background-removal).

## Changes

- CLI
- outputs a mask file (`npy` a Matrix of `uint8` numbers, or an image as `jpg`, `png`)
    - low number = background
    - high = foreground (face, body)

## 概要

顔写真の全面 (顔・身体部分) と背景とを分離するマスクを推定する。

## Usage

```sh
$ cat ./filelist
/file/path/1.jpg /file/path/output/1.npy
/file/path/2.jpg /file/path/output/2.npy
/file/path/3.jpg /file/path/output/3.jpg
/file/path/3.jpg /file/path/output/3.png

$ python3 ./main.py ./filelist
```
