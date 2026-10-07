# Test fixtures

`sample.mp4` is a 2-second synthetic test-pattern clip generated with:

    ffmpeg -y -f lavfi -i testsrc=duration=2:size=320x240:rate=15 -c:v libx264 -pix_fmt yuv420p sample.mp4

Regenerate if the fixture goes missing. No real evidentiary content is stored in this repo.
