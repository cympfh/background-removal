from PIL import Image
import numpy
import click
from background_removal import predict1


def save_mask(src, dst):
    img = Image.open(src)
    mask = predict1(numpy.array(img))
    ext = dst.split('.')[-1]
    if ext == 'npy':
        numpy.save(dst, mask)
    elif ext == 'jpg' or ext == 'png':
        img = Image.fromarray(mask)
        img.save(dst)
    else:
        raise Exception(f"Unknown extention: {ext}")


@click.command()
@click.argument('list_file', type=click.File('r'))
def main(list_file):
    for line in list_file:
        src, dst = line.strip().split()
        print(f"Processing {src} => {dst}")
        save_mask(src, dst)


main()
