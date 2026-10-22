"""Make a matched-time comparison grid from the existing walk replay MP4.

Decode the supplied video itself, verify its existing manifest hash, then crop
the two 384px panels. No generated frames, interpolation, or image retouching.
"""

import argparse
import base64
import hashlib
import html
import io
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
BOXES = {'baseline': (18, 111, 402, 495), 'PPD-KL': (418, 111, 802, 495)}
COLORS = {'baseline': '#4b5563', 'PPD-KL': '#c2410c'}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pixel_hash(array):
    return hashlib.sha256(array.tobytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--title', default='quadruped_walk')
    parser.add_argument('--columns', type=int, choices=(10, 11), default=10)
    parser.add_argument('--compact', action='store_true',
                        help='Keep only method names and timestamps, without heading/footer')
    parser.add_argument('--hide-seed', action='store_true',
                        help='Remove seed0 from the subtitle while keeping the heading')
    parser.add_argument('--no-footer', action='store_true')
    parser.add_argument('--horizontal-gap', type=int, default=10)
    args = parser.parse_args()
    if args.horizontal_gap < 0:
        parser.error('--horizontal-gap must be nonnegative')

    import imageio_ffmpeg
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    video, output = args.video.resolve(), args.output.resolve()
    assert video.name == 'walk_before250k_training_full.mp4'
    manifest_path = video.parent / 'manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    expected_hash = manifest['output_sha256'][video.name]
    assert sha(video) == expected_hash, 'Video differs from its existing manifest'
    assert manifest['encoded_fps'] == 25 and manifest['full_seconds'] == 20
    assert video.name.endswith('_full.mp4'), 'Use the complete original illustration'
    output.mkdir(parents=True, exist_ok=False)

    # 500 decoded frames, from time 0.00 through 19.96 seconds. Round uniform
    # positions to the nearest existing frame, rather than interpolating images.
    columns = args.columns
    if columns == 11:
        nominal_seconds = list(range(0, 21, 2))
        indices = [min(second * 25, 499) for second in nominal_seconds]
        sampling = '0,2,...,20 seconds; use existing frame at each target, clamping end to frame499/19.96s'
    else:
        indices = [round(i * 499 / 9) for i in range(10)]
        nominal_seconds = [index / 25 for index in indices]
        sampling = 'round(i * (500-1) / (10-1)), i=0..9; nearest existing video frame'
    assert len(set(indices)) == columns and indices[0] == 0 and indices[-1] == 499
    selected, selected_hashes = {}, {}
    reader = imageio_ffmpeg.read_frames(str(video), pix_fmt='rgb24')
    metadata = next(reader)
    assert tuple(metadata['size']) == (820, 558) and metadata['fps'] == 25
    count = 0
    for index, raw in enumerate(reader):
        count += 1
        if index in indices:
            frame = np.frombuffer(raw, dtype=np.uint8).reshape(558, 820, 3)
            selected[index] = {label: Image.fromarray(frame[y0:y1, x0:x1].copy())
                               for label, (x0, y0, x1, y1) in BOXES.items()}
            selected_hashes[index] = pixel_hash(frame)
    assert count == 500 and set(selected) == set(indices)

    margin, label_width, cell, gap, row_gap = 24, 170, 384, args.horizontal_gap, 24
    show_heading = not args.compact
    show_footer = not args.compact and not args.no_footer
    top, bottom = (154 if show_heading else 52), (64 if show_footer else 8)
    width = margin * 2 + label_width + columns * cell + (columns-1) * gap
    height = top + 2 * cell + row_gap + bottom
    canvas = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(canvas)
    font_file = 'C:/Windows/Fonts/arial.ttf'
    font = ImageFont.truetype(font_file, 27)
    label_font = ImageFont.truetype(font_file, 30)
    title_font = ImageFont.truetype(font_file, 38)
    small_font = ImageFont.truetype(font_file, 24)
    heading = args.title
    subtitle = ('seed0 | ' if not args.hide_seed else '') + 'training replay, 224k-240k actions'
    footer = 'Matched video times; original MP4 panels, with no frame interpolation.'
    if show_heading:
        draw.text((width/2, 30), heading, font=title_font, anchor='mm', fill='#111111')
        draw.text((width/2, 78), subtitle, font=small_font, anchor='mm', fill='#555555')
    if show_footer:
        draw.text((width/2, height-25), footer, font=small_font, anchor='mm', fill='#555555')

    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
           '<rect width="100%" height="100%" fill="white"/>']

    def svg_text(x, y, value, size, color='#111111', anchor='middle'):
        svg.append(f'<text x="{x}" y="{y}" font-family="Arial, sans-serif" '
                   f'font-size="{size}" fill="{color}" text-anchor="{anchor}" '
                   f'dominant-baseline="central">{html.escape(value)}</text>')

    if show_heading:
        svg_text(width/2, 30, heading, 38)
        svg_text(width/2, 78, subtitle, 24, '#555555')
    if show_footer:
        svg_text(width/2, height-25, footer, 24, '#555555')
    receipts = []
    for column, index in enumerate(indices):
        x = margin + label_width + column * (cell + gap)
        time = index / 25
        nominal = nominal_seconds[column]
        if columns == 11:
            timestamp = ('≈' if time != nominal else '') + f'{nominal} s'
        else:
            timestamp = f'{time:.2f} s'
        timestamp_y = top - 26
        draw.text((x+cell/2, timestamp_y), timestamp, font=font, anchor='mm', fill='#333333')
        svg_text(x+cell/2, timestamp_y, timestamp, 27, '#333333')
        receipt = dict(column=column+1, frame_index=index, seconds=time,
                       nominal_seconds=nominal, display_timestamp=timestamp,
                       decoded_frame_rgb_sha256=selected_hashes[index], panels={})
        for row, label in enumerate(BOXES):
            y = top + row * (cell + row_gap)
            tile = selected[index][label]
            assert tile.size == (cell, cell)
            canvas.paste(tile, (x, y))
            stream = io.BytesIO(); tile.save(stream, format='PNG')
            encoded = base64.b64encode(stream.getvalue()).decode('ascii')
            svg.append(f'<image x="{x}" y="{y}" width="{cell}" height="{cell}" '
                       f'href="data:image/png;base64,{encoded}"/>')
            folder = output / ('baseline' if row == 0 else 'ppd_kl')
            folder.mkdir(exist_ok=True)
            file = folder / f'frame_{index:05d}.png'
            file.write_bytes(stream.getvalue())
            # Cropped pixels must appear unchanged in both the saved panel and grid.
            pixels = np.asarray(tile)
            assert np.array_equal(np.asarray(Image.open(file)), pixels)
            assert np.array_equal(np.asarray(canvas.crop((x,y,x+cell,y+cell))), pixels)
            receipt['panels'][label] = dict(file=str(file.relative_to(output)),
                crop_box=BOXES[label], png_sha256=sha(file),
                decoded_crop_rgb_sha256=pixel_hash(pixels), row=row+1)
        receipts.append(receipt)
    for row, label in enumerate(BOXES):
        y = top + row * (cell + row_gap) + cell/2
        draw.text((margin+label_width-14, y), label, font=label_font,
                  anchor='rm', fill=COLORS[label])
        svg_text(margin+label_width-14, y, label, 30, COLORS[label], 'end')
    svg.append('</svg>')
    png = output / f'walk_comparison_2x{columns}.png'
    vector = output / f'walk_comparison_2x{columns}.svg'
    canvas.save(png, dpi=(300, 300))
    vector.write_text('\n'.join(svg)+'\n', encoding='utf-8')
    # Check the delivered PNG after encoding, not only the in-memory composition.
    with Image.open(png) as delivered:
        assert delivered.size == (width, height)
        for column, index in enumerate(indices):
            x = margin+label_width+column*(cell+gap)
            for row, label in enumerate(BOXES):
                y = top+row*(cell+row_gap)
                assert np.array_equal(np.asarray(delivered.crop((x,y,x+cell,y+cell))),
                                      np.asarray(selected[index][label]))
    assert sha(video) == expected_hash, 'Original video changed during extraction'
    record = dict(git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        script_sha256=sha(Path(__file__)), source_video=str(video), source_video_sha256=expected_hash,
        source_manifest_sha256=sha(manifest_path), decoded_video=metadata,
        decoded_frame_count=count, sampling=sampling,
        row_order=list(BOXES), columns=columns, total_panels=2*columns, output_size=[width,height],
        nominal_seconds=nominal_seconds, exact_seconds=[i/25 for i in indices],
        horizontal_gap_pixels=gap, vertical_row_gap_pixels=row_gap,
        compact=args.compact, heading_and_footer_visible=show_heading and show_footer,
        heading_visible=show_heading, footer_visible=show_footer,
        seed_label_visible=show_heading and not args.hide_seed,
        heading=heading if show_heading else None, subtitle=subtitle if show_heading else None,
        evidence_kind=manifest['evidence_kind'], actual_training_action_range=[224001,239985],
        same_video_frame_in_each_column=True, no_frame_interpolation=True,
        source_unchanged=True, all_png_panels_pixel_identical_to_decoded_crops=True,
        ffmpeg_version=imageio_ffmpeg.get_ffmpeg_version(), frames=receipts,
        output_sha256={str(p.relative_to(output)):sha(p) for p in output.rglob('*') if p.is_file()})
    (output/'sampling.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(output=str(output),shape=f'2 rows x {columns} columns',
                          timestamps=[i/25 for i in indices], source_unchanged=True)))


if __name__ == '__main__':
    main()
