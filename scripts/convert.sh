convert rose: -crop 20x20 \
          -set filename:tile "%[fx:page.x/20+1]_%[fx:page.y/20+1]" \
          +repage +adjoin "rose_tile_%[filename:tile].gif"

convert MD657-N48-2017.02.22-16.41.55_MD657_2_0143_lossless.tif -crop 1000x1000+100+100  +repage new_tiles/tiles_%02d.tif
