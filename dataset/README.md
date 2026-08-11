# Dataset directory

Place the ASL Alphabet image dataset at `dataset/asl_alphabet_train/`, with one
directory per label (`A/`, `B/`, `space/`, `del/`, and `nothing/`). The raw
dataset and generated landmark cache are ignored by Git because they are large.

The same directory supplies representative fingerspelling images by default.
Set `ASL_SIGNS_DIR` to a smaller directory with the same layout in production.
