import argparse
import glob
import os
import random
import sys

import numpy as np

import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, Conv2DTranspose, concatenate,
    BatchNormalization, Dropout, Activation
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    ModelCheckpoint, CSVLogger, ReduceLROnPlateau, EarlyStopping
)
from tensorflow.keras.utils import Sequence, to_categorical
from sklearn.model_selection import train_test_split

SEED = 42
N_CLASSES = 4  
CLASS_NAMES = ["background", "necrotic_core", "edema", "enhancing_tumor"]
MODALITY_FILES = {
    "flair": "_flair.nii",
    "t1": "_t1.nii",
    "t1ce": "_t1ce.nii",
    "t2": "_t2.nii",
    "seg": "_seg.nii",
}


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def _find_modality_file(patient_dir, pid, suffix):
    """Find a modality file, tolerating both .nii and .nii.gz extensions."""
    for ext in ("", ".gz"):
        path = os.path.join(patient_dir, f"{pid}{suffix}{ext}")
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        f"Could not find {pid}{suffix}[.gz] under {patient_dir}"
    )


def list_patient_dirs(data_root):
    dirs = sorted(
        d for d in glob.glob(os.path.join(data_root, "*"))
        if os.path.isdir(d) and not os.path.basename(d).startswith(".")
    )
    if not dirs:
        raise RuntimeError(
            f"No patient folders found under {data_root}. "
            "Check --data-root points at the folder containing "
            "BraTS20_Training_XXX subfolders."
        )
    return dirs


def load_nifti(path):
    import nibabel as nib
    return nib.load(path).get_fdata()


def load_patient_volumes(patient_dir, modalities=("flair", "t1ce"), include_seg=True):
    """Load requested modality volumes (+ segmentation) for one patient."""
    pid = os.path.basename(patient_dir)
    volumes = {}
    for mod in modalities:
        path = _find_modality_file(patient_dir, pid, MODALITY_FILES[mod])
        volumes[mod] = load_nifti(path)
    if include_seg:
        seg_path = _find_modality_file(patient_dir, pid, MODALITY_FILES["seg"])
        volumes["seg"] = load_nifti(seg_path)
    return volumes


def remap_mask(mask_slice):
    """BraTS raw labels {0,1,2,4} -> contiguous {0,1,2,3}."""
    mask_slice = mask_slice.copy()
    mask_slice[mask_slice == 4] = 3
    return mask_slice


def normalize_slice(x):
    mx = x.max()
    return x / mx if mx > 0 else x


def resize_slice(x, size, is_mask=False):
    from skimage.transform import resize
    if is_mask:
        return resize(x, (size, size), preserve_range=True, order=0, anti_aliasing=False)
    return resize(x, (size, size), preserve_range=True)



class BraTSSequence(Sequence):
    """
    Streams 2D axial slices patient-by-patient so the full dataset never
    needs to fit in memory at once.
    """

    def __init__(self, patient_dirs, modalities=("flair", "t1ce"), batch_size=8,
                 img_size=128, volume_slices=100, start_slice=22, shuffle=True):
        self.patient_dirs = patient_dirs
        self.modalities = modalities
        self.batch_size = batch_size
        self.img_size = img_size
        self.volume_slices = volume_slices
        self.start_slice = start_slice
        self.shuffle = shuffle
        self.n_channels = len(modalities)
        self.index_pairs = [
            (p_idx, s) for p_idx in range(len(patient_dirs))
            for s in range(start_slice, start_slice + volume_slices)
        ]
        self.on_epoch_end()

    def __len__(self):
        return max(1, len(self.index_pairs) // self.batch_size)

    def on_epoch_end(self):
        if self.shuffle:
            random.shuffle(self.index_pairs)

    def __getitem__(self, idx):
        batch_pairs = self.index_pairs[idx * self.batch_size:(idx + 1) * self.batch_size]
        X = np.zeros((self.batch_size, self.img_size, self.img_size, self.n_channels), dtype=np.float32)
        y = np.zeros((self.batch_size, self.img_size, self.img_size, N_CLASSES), dtype=np.float32)

        cache = {}
        for i, (p_idx, s) in enumerate(batch_pairs):
            pdir = self.patient_dirs[p_idx]
            if p_idx not in cache:
                cache[p_idx] = load_patient_volumes(pdir, modalities=self.modalities, include_seg=True)
            vols = cache[p_idx]

            for c, mod in enumerate(self.modalities):
                sl = resize_slice(vols[mod][:, :, s], self.img_size)
                X[i, :, :, c] = normalize_slice(sl)

            seg_sl = resize_slice(vols["seg"][:, :, s], self.img_size, is_mask=True)
            seg_sl = remap_mask(np.round(seg_sl))
            y[i] = to_categorical(seg_sl, num_classes=N_CLASSES)

        return X, y


def conv_block(x, filters, dropout_rate=0.0):
    x = Conv2D(filters, 3, padding="same", kernel_initializer="he_normal")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv2D(filters, 3, padding="same", kernel_initializer="he_normal")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    if dropout_rate > 0:
        x = Dropout(dropout_rate)(x)
    return x


def build_unet(input_shape, n_classes=N_CLASSES, base_filters=32):
    inputs = Input(input_shape)

    c1 = conv_block(inputs, base_filters)
    p1 = MaxPooling2D()(c1)

    c2 = conv_block(p1, base_filters * 2)
    p2 = MaxPooling2D()(c2)

    c3 = conv_block(p2, base_filters * 4)
    p3 = MaxPooling2D()(c3)

    c4 = conv_block(p3, base_filters * 8, dropout_rate=0.3)
    p4 = MaxPooling2D()(c4)

    bn = conv_block(p4, base_filters * 16, dropout_rate=0.4)

    u4 = Conv2DTranspose(base_filters * 8, 2, strides=2, padding="same")(bn)
    u4 = concatenate([u4, c4])
    c5 = conv_block(u4, base_filters * 8, dropout_rate=0.3)

    u3 = Conv2DTranspose(base_filters * 4, 2, strides=2, padding="same")(c5)
    u3 = concatenate([u3, c3])
    c6 = conv_block(u3, base_filters * 4)

    u2 = Conv2DTranspose(base_filters * 2, 2, strides=2, padding="same")(c6)
    u2 = concatenate([u2, c2])
    c7 = conv_block(u2, base_filters * 2)

    u1 = Conv2DTranspose(base_filters, 2, strides=2, padding="same")(c7)
    u1 = concatenate([u1, c1])
    c8 = conv_block(u1, base_filters)

    outputs = Conv2D(n_classes, 1, activation="softmax")(c8)
    return Model(inputs, outputs, name="UNet_BraTS")


def dice_coef(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (
        tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth
    )


def dice_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)


def combined_loss(y_true, y_pred):
    cce = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
    return cce + dice_loss(y_true, y_pred)


def make_dice_coef_class(class_idx):
    def metric(y_true, y_pred):
        return dice_coef(y_true[..., class_idx], y_pred[..., class_idx])
    metric.__name__ = f"dice_class_{class_idx}"
    return metric


def get_custom_objects():
    custom = {
        "combined_loss": combined_loss,
        "dice_loss": dice_loss,
        "dice_coef": dice_coef,
    }
    for i in range(N_CLASSES):
        fn = make_dice_coef_class(i)
        custom[fn.__name__] = fn
    return custom

def cmd_train(args):
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    ckpt_dir = os.path.join(args.output_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    modalities = tuple(m.strip() for m in args.modalities.split(","))
    print(f"Using modalities: {modalities}")

    patient_dirs = list_patient_dirs(args.data_root)
    print(f"Found {len(patient_dirs)} patients")

    train_dirs, val_dirs = train_test_split(
        patient_dirs, test_size=args.val_split, random_state=args.seed
    )
    print(f"Train: {len(train_dirs)} patients | Val: {len(val_dirs)} patients")

    train_gen = BraTSSequence(
        train_dirs, modalities=modalities, batch_size=args.batch_size,
        img_size=args.img_size, volume_slices=args.volume_slices,
        start_slice=args.start_slice, shuffle=True,
    )
    val_gen = BraTSSequence(
        val_dirs, modalities=modalities, batch_size=args.batch_size,
        img_size=args.img_size, volume_slices=args.volume_slices,
        start_slice=args.start_slice, shuffle=False,
    )
    print(f"Train batches/epoch: {len(train_gen)} | Val batches/epoch: {len(val_gen)}")

    input_shape = (args.img_size, args.img_size, len(modalities))
    model = build_unet(input_shape, base_filters=args.base_filters)
    model.summary()

    metrics = [dice_coef] + [make_dice_coef_class(i) for i in range(1, N_CLASSES)] + ["accuracy"]
    model.compile(optimizer=Adam(learning_rate=args.lr), loss=combined_loss, metrics=metrics)

    callbacks = [
        ModelCheckpoint(
            os.path.join(ckpt_dir, "unet_brats_best.keras"),
            monitor="val_dice_coef", mode="max", save_best_only=True, verbose=1,
        ),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1),
        EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True, verbose=1),
        CSVLogger(os.path.join(ckpt_dir, "training_log.csv")),
    ]

    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=1,
    )

    final_path = os.path.join(ckpt_dir, "unet_brats_final.keras")
    model.save(final_path)
    print(f"\nTraining complete. Final model saved to: {final_path}")
    print(f"Best checkpoint (by val_dice_coef) saved to: {os.path.join(ckpt_dir, 'unet_brats_best.keras')}")


def cmd_evaluate(args):
    set_seed(args.seed)
    modalities = tuple(m.strip() for m in args.modalities.split(","))
    patient_dirs = list_patient_dirs(args.data_root)
    _, val_dirs = train_test_split(patient_dirs, test_size=args.val_split, random_state=args.seed)

    val_gen = BraTSSequence(
        val_dirs, modalities=modalities, batch_size=args.batch_size,
        img_size=args.img_size, volume_slices=args.volume_slices,
        start_slice=args.start_slice, shuffle=False,
    )

    print(f"Loading model from {args.model_path} ...")
    model = load_model(args.model_path, custom_objects=get_custom_objects())

    print(f"Evaluating on {len(val_dirs)} validation patients ...")
    results = model.evaluate(val_gen, verbose=1)
    print("\nValidation results:")
    for name, value in zip(model.metrics_names, results):
        print(f"  {name}: {value:.4f}")


def _predict_slice(model, patient_dir, modalities, img_size, slice_idx):
    pid = os.path.basename(patient_dir)
    vols = load_patient_volumes(patient_dir, modalities=modalities, include_seg=True)

    X = np.zeros((1, img_size, img_size, len(modalities)), dtype=np.float32)
    for c, mod in enumerate(modalities):
        sl = resize_slice(vols[mod][:, :, slice_idx], img_size)
        X[0, :, :, c] = normalize_slice(sl)

    seg_sl = remap_mask(np.round(resize_slice(vols["seg"][:, :, slice_idx], img_size, is_mask=True)))
    pred = model.predict(X, verbose=0)[0]
    pred_mask = np.argmax(pred, axis=-1)

    return pid, vols, X, seg_sl, pred_mask


def cmd_predict(args):
    modalities = tuple(m.strip() for m in args.modalities.split(","))
    patient_dir = os.path.join(args.data_root, args.patient_id)
    if not os.path.isdir(patient_dir):
        raise RuntimeError(f"Patient folder not found: {patient_dir}")

    print(f"Loading model from {args.model_path} ...")
    model = load_model(args.model_path, custom_objects=get_custom_objects())

    pid, vols, X, seg_sl, pred_mask = _predict_slice(
        model, patient_dir, modalities, args.img_size, args.slice_idx
    )

    import matplotlib.pyplot as plt
    n_panels = len(modalities) + 2
    fig, axes = plt.subplots(1, n_panels, figsize=(4 * n_panels, 4))
    for c, mod in enumerate(modalities):
        axes[c].imshow(X[0, :, :, c], cmap="gray")
        axes[c].set_title(mod.upper())
        axes[c].axis("off")
    axes[-2].imshow(seg_sl, cmap="nipy_spectral", vmin=0, vmax=N_CLASSES - 1)
    axes[-2].set_title("Ground truth")
    axes[-2].axis("off")
    axes[-1].imshow(pred_mask, cmap="nipy_spectral", vmin=0, vmax=N_CLASSES - 1)
    axes[-1].set_title("Prediction")
    axes[-1].axis("off")
    plt.suptitle(f"{pid} — slice {args.slice_idx}")
    plt.tight_layout()

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, f"{pid}_slice{args.slice_idx}_prediction.png")
    plt.savefig(out_path, dpi=150)
    print(f"Saved visualization to {out_path}")


def cmd_segment_volume(args):
    modalities = tuple(m.strip() for m in args.modalities.split(","))
    patient_dir = os.path.join(args.data_root, args.patient_id)
    if not os.path.isdir(patient_dir):
        raise RuntimeError(f"Patient folder not found: {patient_dir}")

    print(f"Loading model from {args.model_path} ...")
    model = load_model(args.model_path, custom_objects=get_custom_objects())

    vols = load_patient_volumes(patient_dir, modalities=modalities, include_seg=False)
    ref_shape = vols[modalities[0]].shape
    depth = ref_shape[2]
    full_pred = np.zeros(ref_shape, dtype=np.uint8)

    print(f"Segmenting {depth} slices for {args.patient_id} ...")
    for s in range(depth):
        X = np.zeros((1, args.img_size, args.img_size, len(modalities)), dtype=np.float32)
        for c, mod in enumerate(modalities):
            sl = resize_slice(vols[mod][:, :, s], args.img_size)
            X[0, :, :, c] = normalize_slice(sl)
        pred = model.predict(X, verbose=0)[0]
        pred_mask = np.argmax(pred, axis=-1).astype(np.uint8)
        pred_mask_full = resize_slice(pred_mask, ref_shape[0], is_mask=True).astype(np.uint8)
        # resize_slice assumes square target from img_size arg; handle non-square ref_shape safely
        if ref_shape[0] != ref_shape[1]:
            from skimage.transform import resize
            pred_mask_full = resize(
                pred_mask, (ref_shape[0], ref_shape[1]),
                preserve_range=True, order=0, anti_aliasing=False
            ).astype(np.uint8)
        full_pred[:, :, s] = pred_mask_full

    import nibabel as nib
    os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)
    nib.save(nib.Nifti1Image(full_pred, np.eye(4)), args.output_file)
    print(f"Saved full 3D segmentation to {args.output_file}")

def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Brain Tumor Segmentation — 2D U-Net on BraTS (MRI)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(p, need_model=False):
        p.add_argument("--data-root", required=True, help="Path to folder containing patient subfolders")
        p.add_argument("--output-dir", default="./output", help="Where to save checkpoints/outputs")
        p.add_argument("--img-size", type=int, default=128, help="Slice resize resolution (square)")
        p.add_argument("--batch-size", type=int, default=8)
        p.add_argument("--volume-slices", type=int, default=100, help="Number of axial slices to use per patient")
        p.add_argument("--start-slice", type=int, default=22, help="First axial slice index to use")
        p.add_argument("--val-split", type=float, default=0.15)
        p.add_argument("--modalities", default="flair,t1ce",
                        help="Comma-separated modalities to use as input channels, "
                             "e.g. 'flair,t1ce' or 'flair,t1,t1ce,t2'")
        p.add_argument("--seed", type=int, default=SEED)
        if need_model:
            p.add_argument("--model-path", required=True, help="Path to a saved .keras model")

    p_train = sub.add_parser("train", help="Train a new U-Net model")
    add_common(p_train)
    p_train.add_argument("--epochs", type=int, default=30)
    p_train.add_argument("--lr", type=float, default=1e-4)
    p_train.add_argument("--base-filters", type=int, default=32, help="Base channel width of the U-Net")
    p_train.set_defaults(func=cmd_train)

    p_eval = sub.add_parser("evaluate", help="Evaluate a saved model on the validation split")
    add_common(p_eval, need_model=True)
    p_eval.set_defaults(func=cmd_evaluate)

    p_pred = sub.add_parser("predict", help="Run inference on one slice and save a visualization PNG")
    add_common(p_pred, need_model=True)
    p_pred.add_argument("--patient-id", required=True, help="Patient folder name, e.g. BraTS20_Training_001")
    p_pred.add_argument("--slice-idx", type=int, default=90)
    p_pred.set_defaults(func=cmd_predict)

    p_seg = sub.add_parser("segment-volume", help="Segment a full 3D volume and save as NIfTI")
    add_common(p_seg, need_model=True)
    p_seg.add_argument("--patient-id", required=True)
    p_seg.add_argument("--output-file", required=True, help="Output .nii.gz path")
    p_seg.set_defaults(func=cmd_segment_volume)

    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    print(f"GPU available: {tf.config.list_physical_devices('GPU')}")
    args.func(args)


if __name__ == "__main__":
    main()