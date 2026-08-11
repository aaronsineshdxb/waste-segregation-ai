#!/usr/bin/env python3
"""
Model Training Script for Waste Segregation Assistant

Trains a CNN classifier for waste categorization (recyclable, compost, landfill)
using transfer learning with pre-trained models.

Usage:
    python scripts/train_model.py --config config/training_config.yaml
    python scripts/train_model.py --epochs 30 --batch-size 32 --base-model EfficientNetB0
"""

import argparse
import json
from pathlib import Path

import numpy as np
import yaml
from tensorflow import keras
from tensorflow.keras import applications, callbacks, layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


class TrainingConfig:
    """Configuration for training."""

    def __init__(self, config_path=None):
        # Defaults
        self.data_dir = "data/raw"
        self.model_dir = "models"
        self.img_size = (224, 224)
        self.batch_size = 32
        self.categories = ["recyclable", "compost", "landfill"]
        self.num_classes = 3
        self.base_model_name = "MobileNetV2"
        self.epochs = 30
        self.fine_tune_epochs = 15
        self.initial_lr = 1e-3
        self.fine_tune_lr = 1e-5
        self.validation_split = 0.2
        self.use_pretrained = True
        self.seed = 42

        if config_path and Path(config_path).exists():
            with open(config_path) as f:
                cfg = yaml.safe_load(f)
            for k, v in cfg.items():
                if hasattr(self, k):
                    setattr(self, k, v)


class WasteClassifier:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.data_dir = Path(config.data_dir)
        self.model_dir = Path(config.model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.img_size = config.img_size
        self.batch_size = config.batch_size
        self.categories = config.categories
        self.num_classes = config.num_classes
        self.model = None
        self.history = None
        self.train_gen = None
        self.val_gen = None

    def create_data_generators(self):
        """Create training and validation data generators with augmentation."""
        train_datagen = ImageDataGenerator(
            rescale=1.0 / 255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            brightness_range=[0.8, 1.2],
            fill_mode="nearest",
            validation_split=self.config.validation_split,
        )

        val_datagen = ImageDataGenerator(
            rescale=1.0 / 255,
            validation_split=self.config.validation_split,
        )

        self.train_gen = train_datagen.flow_from_directory(
            self.data_dir,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode="categorical",
            subset="training",
            shuffle=True,
            seed=self.config.seed,
        )

        self.val_gen = val_datagen.flow_from_directory(
            self.data_dir,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode="categorical",
            subset="validation",
            shuffle=False,
            seed=self.config.seed,
        )

        print(f"Training samples: {self.train_gen.samples}")
        print(f"Validation samples: {self.val_gen.samples}")
        print(f"Classes: {list(self.train_gen.class_indices.keys())}")

        # Check class balance
        class_counts = np.bincount(self.train_gen.classes)
        print(f"Class distribution: {dict(zip(self.categories, class_counts))}")

        return self.train_gen, self.val_gen

    def build_model(self):
        """Build transfer learning model using pre-trained base."""
        weights = "imagenet" if self.config.use_pretrained else None

        if self.config.base_model_name == "MobileNetV2":
            base_model = applications.MobileNetV2(
                weights=weights,
                include_top=False,
                input_shape=(*self.img_size, 3),
            )
        elif self.config.base_model_name == "EfficientNetB0":
            base_model = applications.EfficientNetB0(
                weights=weights,
                include_top=False,
                input_shape=(*self.img_size, 3),
            )
        elif self.config.base_model_name == "ResNet50":
            base_model = applications.ResNet50(
                weights=weights,
                include_top=False,
                input_shape=(*self.img_size, 3),
            )
        else:
            raise ValueError(f"Unsupported base model: {self.config.base_model_name}")

        # Freeze base model initially
        base_model.trainable = False

        # Build the model
        inputs = keras.Input(shape=(*self.img_size, 3))
        x = base_model(inputs, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(256, activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)
        outputs = layers.Dense(self.num_classes, activation="softmax")(x)

        self.model = keras.Model(inputs, outputs)

        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.config.initial_lr),
            loss="categorical_crossentropy",
            metrics=[
                "accuracy",
                keras.metrics.Precision(name="precision"),
                keras.metrics.Recall(name="recall"),
                keras.metrics.AUC(name="auc"),
            ],
        )

        print(self.model.summary())
        return self.model

    def get_callbacks(self):
        """Get training callbacks."""
        return [
            callbacks.ModelCheckpoint(
                self.model_dir / "best_model.keras",
                monitor="val_accuracy",
                save_best_only=True,
                mode="max",
                verbose=1,
            ),
            callbacks.EarlyStopping(
                monitor="val_accuracy",
                patience=8,
                restore_best_weights=True,
                verbose=1,
            ),
            callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.2,
                patience=4,
                min_lr=1e-7,
                verbose=1,
            ),
            callbacks.CSVLogger(self.model_dir / "training_log.csv"),
            callbacks.TensorBoard(
                log_dir=str(self.model_dir / "logs"),
                histogram_freq=1,
                write_graph=True,
            ),
        ]

    def train(self):
        """Train the model with transfer learning and fine-tuning."""
        if self.train_gen is None or self.val_gen is None:
            self.create_data_generators()

        if self.model is None:
            self.build_model()

        cb = self.get_callbacks()

        # Phase 1: Train top layers only
        print("\n" + "=" * 60)
        print(
            f"PHASE 1: Training top layers (frozen base) - {self.config.epochs} epochs"
        )
        print("=" * 60)

        self.history = self.model.fit(
            self.train_gen,
            epochs=self.config.epochs,
            validation_data=self.val_gen,
            callbacks=cb,
            verbose=1,
        )

        # Phase 2: Fine-tuning
        print("\n" + "=" * 60)
        print(
            f"PHASE 2: Fine-tuning (unfreeze base) - {self.config.fine_tune_epochs} epochs"
        )
        print("=" * 60)

        # Unfreeze base model
        self.model.layers[1].trainable = True

        # Optionally freeze early layers for stability
        # Unfreeze only the last N layers of base model
        base_model = self.model.layers[1]
        for layer in base_model.layers[:-20]:
            layer.trainable = False

        # Recompile with lower learning rate
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.config.fine_tune_lr),
            loss="categorical_crossentropy",
            metrics=[
                "accuracy",
                keras.metrics.Precision(name="precision"),
                keras.metrics.Recall(name="recall"),
                keras.metrics.AUC(name="auc"),
            ],
        )

        # Continue training
        fine_tune_history = self.model.fit(
            self.train_gen,
            epochs=self.config.epochs + self.config.fine_tune_epochs,
            initial_epoch=self.config.epochs,
            validation_data=self.val_gen,
            callbacks=cb,
            verbose=1,
        )

        # Combine histories
        for key in self.history.history:
            self.history.history[key].extend(fine_tune_history.history[key])

        # Save final model
        self.model.save(self.model_dir / "final_model.keras")
        print(f"\nModel saved to {self.model_dir / 'final_model.keras'}")

        # Save class indices
        with open(self.model_dir / "class_indices.json", "w") as f:
            json.dump(self.train_gen.class_indices, f, indent=2)

        # Save config
        with open(self.model_dir / "training_config.json", "w") as f:
            json.dump(vars(self.config), f, indent=2, default=str)

        return self.history

    def evaluate(self):
        """Evaluate model on validation set."""
        if self.val_gen is None:
            self.create_data_generators()

        if self.model is None:
            model_path = self.model_dir / "best_model.keras"
            if model_path.exists():
                self.model = keras.models.load_model(model_path)
                print(f"Loaded model from {model_path}")
            else:
                print(f"Model not found at {model_path}")
                return None

        print("\n" + "=" * 60)
        print("EVALUATION")
        print("=" * 60)

        # Get predictions
        self.val_gen.reset()
        predictions = self.model.predict(self.val_gen, verbose=1)
        y_pred = np.argmax(predictions, axis=1)
        y_true = self.val_gen.classes

        # Classification report
        print("\nClassification Report:")
        report_str = classification_report(y_true, y_pred, target_names=self.categories)
        print(report_str)

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=self.categories,
            yticklabels=self.categories,
        )
        plt.title("Confusion Matrix")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.tight_layout()
        plt.savefig(self.model_dir / "confusion_matrix.png", dpi=300)
        plt.close()

        # Per-class metrics
        report = classification_report(
            y_true, y_pred, target_names=self.categories, output_dict=True
        )
        with open(self.model_dir / "evaluation_report.json", "w") as f:
            json.dump(report, f, indent=2)

        # Print summary
        print(f"\nOverall Accuracy: {report['accuracy']:.4f}")
        print(f"Macro F1: {report['macro avg']['f1-score']:.4f}")
        print(f"Weighted F1: {report['weighted avg']['f1-score']:.4f}")

        return report

    def plot_training_history(self):
        """Plot training history."""
        if self.history is None:
            print("No training history available")
            return

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))

        # Accuracy
        axes[0, 0].plot(
            self.history.history["accuracy"], label="Train", alpha=0.8
        )
        axes[0, 0].plot(
            self.history.history["val_accuracy"], label="Validation", alpha=0.8
        )
        axes[0, 0].axvline(
            x=self.config.epochs - 0.5, color="red", linestyle="--", label="Fine-tune start"
        )
        axes[0, 0].set_title("Model Accuracy")
        axes[0, 0].set_xlabel("Epoch")
        axes[0, 0].set_ylabel("Accuracy")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Loss
        axes[0, 1].plot(self.history.history["loss"], label="Train", alpha=0.8)
        axes[0, 1].plot(self.history.history["val_loss"], label="Validation", alpha=0.8)
        axes[0, 1].axvline(
            x=self.config.epochs - 0.5, color="red", linestyle="--", label="Fine-tune start"
        )
        axes[0, 1].set_title("Model Loss")
        axes[0, 1].set_xlabel("Epoch")
        axes[0, 1].set_ylabel("Loss")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Precision
        if "precision" in self.history.history:
            axes[0, 2].plot(
                self.history.history["precision"], label="Train", alpha=0.8
            )
            axes[0, 2].plot(
                self.history.history["val_precision"], label="Validation", alpha=0.8
            )
            axes[0, 2].axvline(
                x=self.config.epochs - 0.5, color="red", linestyle="--"
            )
            axes[0, 2].set_title("Precision")
            axes[0, 2].set_xlabel("Epoch")
            axes[0, 2].set_ylabel("Precision")
            axes[0, 2].legend()
            axes[0, 2].grid(True, alpha=0.3)

        # Recall
        if "recall" in self.history.history:
            axes[1, 0].plot(self.history.history["recall"], label="Train", alpha=0.8)
            axes[1, 0].plot(
                self.history.history["val_recall"], label="Validation", alpha=0.8
            )
            axes[1, 0].axvline(
                x=self.config.epochs - 0.5, color="red", linestyle="--"
            )
            axes[1, 0].set_title("Recall")
            axes[1, 0].set_xlabel("Epoch")
            axes[1, 0].set_ylabel("Recall")
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)

        # AUC
        if "auc" in self.history.history:
            axes[1, 1].plot(self.history.history["auc"], label="Train", alpha=0.8)
            axes[1, 1].plot(
                self.history.history["val_auc"], label="Validation", alpha=0.8
            )
            axes[1, 1].axvline(
                x=self.config.epochs - 0.5, color="red", linestyle="--"
            )
            axes[1, 1].set_title("AUC")
            axes[1, 1].set_xlabel("Epoch")
            axes[1, 1].set_ylabel("AUC")
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)

        # Learning rate (if available)
        if "lr" in self.history.history:
            axes[1, 2].plot(self.history.history["lr"], alpha=0.8)
            axes[1, 2].axvline(
                x=self.config.epochs - 0.5, color="red", linestyle="--"
            )
            axes[1, 2].set_title("Learning Rate")
            axes[1, 2].set_xlabel("Epoch")
            axes[1, 2].set_ylabel("LR")
            axes[1, 2].set_yscale("log")
            axes[1, 2].grid(True, alpha=0.3)
        else:
            axes[1, 2].axis("off")

        plt.tight_layout()
        plt.savefig(self.model_dir / "training_history.png", dpi=300)
        plt.close()
        print(f"Training plots saved to {self.model_dir}")


def main():
    parser = argparse.ArgumentParser(description="Train waste classification model")
    parser.add_argument(
        "--config", default="config/training_config.yaml", help="Config file path"
    )
    parser.add_argument("--data-dir", default="data/raw", help="Data directory")
    parser.add_argument("--model-dir", default="models", help="Model output directory")
    parser.add_argument("--epochs", type=int, default=30, help="Initial training epochs")
    parser.add_argument(
        "--fine-tune-epochs", type=int, default=15, help="Fine-tuning epochs"
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument(
        "--base-model",
        default="MobileNetV2",
        choices=["MobileNetV2", "EfficientNetB0", "ResNet50"],
    )
    parser.add_argument(
        "--no-pretrained", action="store_true", help="Don't use ImageNet weights"
    )
    parser.add_argument(
        "--eval-only", action="store_true", help="Only evaluate existing model"
    )
    args = parser.parse_args()

    # Load config
    config = TrainingConfig(args.config if Path(args.config).exists() else None)

    # Override with CLI args
    config.data_dir = args.data_dir
    config.model_dir = args.model_dir
    config.epochs = args.epochs
    config.fine_tune_epochs = args.fine_tune_epochs
    config.batch_size = args.batch_size
    config.base_model_name = args.base_model
    config.use_pretrained = not args.no_pretrained

    classifier = WasteClassifier(config)

    if args.eval_only:
        classifier.evaluate()
    else:
        classifier.train()
        classifier.plot_training_history()
        classifier.evaluate()


if __name__ == "__main__":
    main()
