#!/usr/bin/env python3
"""
Model Training Script for Waste Segregation Assistant
Trains a CNN classifier for waste categorization (recyclable, compost, landfill)
"""

import os
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns


class WasteClassifier:
    def __init__(self, data_dir="data/raw", model_dir="models", img_size=(224, 224), batch_size=32):
        self.data_dir = Path(data_dir)
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.img_size = img_size
        self.batch_size = batch_size
        self.categories = ["recyclable", "compost", "landfill"]
        self.num_classes = len(self.categories)
        self.model = None
        self.history = None
        
    def create_data_generators(self, validation_split=0.2):
        """Create training and validation data generators with augmentation"""
        
        # Training data generator with augmentation
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest',
            validation_split=validation_split
        )
        
        # Validation data generator (only rescaling)
        val_datagen = ImageDataGenerator(
            rescale=1./255,
            validation_split=validation_split
        )
        
        train_generator = train_datagen.flow_from_directory(
            self.data_dir,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            subset='training',
            shuffle=True,
            seed=42
        )
        
        val_generator = val_datagen.flow_from_directory(
            self.data_dir,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            subset='validation',
            shuffle=False,
            seed=42
        )
        
        print(f"Training samples: {train_generator.samples}")
        print(f"Validation samples: {val_generator.samples}")
        print(f"Classes: {list(train_generator.class_indices.keys())}")
        
        return train_generator, val_generator
    
    def build_model(self, base_model_name="MobileNetV2"):
        """Build transfer learning model using pre-trained base"""
        
        # Use weights=None to avoid external downloads/certificate issues on this machine.
        if base_model_name == "MobileNetV2":
            base_model = keras.applications.MobileNetV2(
                weights=None,
                include_top=False,
                input_shape=(*self.img_size, 3)
            )
        elif base_model_name == "EfficientNetB0":
            base_model = keras.applications.EfficientNetB0(
                weights=None,
                include_top=False,
                input_shape=(*self.img_size, 3)
            )
        else:
            raise ValueError(f"Unsupported base model: {base_model_name}")
        
        # Freeze base model initially
        base_model.trainable = False
        
        # Build the model
        inputs = keras.Input(shape=(*self.img_size, 3))
        x = base_model(inputs, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.2)(x)
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        self.model = keras.Model(inputs, outputs)
        
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy', keras.metrics.Precision(name='precision'), 
                    keras.metrics.Recall(name='recall')]
        )
        
        print(self.model.summary())
        return self.model
    
    def train(self, epochs=20, fine_tune_epochs=10):
        """Train the model with transfer learning and fine-tuning"""
        
        train_gen, val_gen = self.create_data_generators()
        
        # Callbacks
        checkpoint = callbacks.ModelCheckpoint(
            self.model_dir / "best_model.keras",
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        )
        
        early_stop = callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=5,
            restore_best_weights=True,
            verbose=1
        )
        
        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=3,
            min_lr=1e-7,
            verbose=1
        )
        
        csv_logger = callbacks.CSVLogger(
            self.model_dir / "training_log.csv"
        )
        
        # Phase 1: Train top layers only
        print("\n" + "="*50)
        print("PHASE 1: Training top layers (frozen base)")
        print("="*50)
        
        self.history = self.model.fit(
            train_gen,
            epochs=epochs,
            validation_data=val_gen,
            callbacks=[checkpoint, early_stop, reduce_lr, csv_logger],
            verbose=1
        )
        
        # Phase 2: Fine-tuning
        print("\n" + "="*50)
        print("PHASE 2: Fine-tuning (unfreeze base)")
        print("="*50)
        
        # Unfreeze base model
        self.model.layers[1].trainable = True
        
        # Recompile with lower learning rate
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-5),
            loss='categorical_crossentropy',
            metrics=['accuracy', keras.metrics.Precision(name='precision'), 
                    keras.metrics.Recall(name='recall')]
        )
        
        # Continue training
        fine_tune_history = self.model.fit(
            train_gen,
            epochs=epochs + fine_tune_epochs,
            initial_epoch=epochs,
            validation_data=val_gen,
            callbacks=[checkpoint, early_stop, reduce_lr, csv_logger],
            verbose=1
        )
        
        # Combine histories
        for key in self.history.history:
            self.history.history[key].extend(fine_tune_history.history[key])
        
        # Save final model
        self.model.save(self.model_dir / "final_model.keras")
        print(f"\nModel saved to {self.model_dir / 'final_model.keras'}")
        
        # Save class indices
        with open(self.model_dir / "class_indices.json", 'w') as f:
            json.dump(train_gen.class_indices, f)
            
        return self.history
    
    def evaluate(self, val_generator=None):
        """Evaluate model on validation set"""
        if val_generator is None:
            _, val_generator = self.create_data_generators()
            
        print("\n" + "="*50)
        print("EVALUATION")
        print("="*50)
        
        # Get predictions
        val_generator.reset()
        predictions = self.model.predict(val_generator, verbose=1)
        y_pred = np.argmax(predictions, axis=1)
        y_true = val_generator.classes
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, target_names=self.categories))
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.categories, yticklabels=self.categories)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(self.model_dir / "confusion_matrix.png", dpi=300)
        plt.close()
        
        # Save metrics
        report = classification_report(y_true, y_pred, target_names=self.categories, output_dict=True)
        with open(self.model_dir / "evaluation_report.json", 'w') as f:
            json.dump(report, f, indent=2)
            
        return report
    
    def plot_training_history(self):
        """Plot training history"""
        if self.history is None:
            print("No training history available")
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Accuracy
        axes[0, 0].plot(self.history.history['accuracy'], label='Train')
        axes[0, 0].plot(self.history.history['val_accuracy'], label='Validation')
        axes[0, 0].set_title('Model Accuracy')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Loss
        axes[0, 1].plot(self.history.history['loss'], label='Train')
        axes[0, 1].plot(self.history.history['val_loss'], label='Validation')
        axes[0, 1].set_title('Model Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Precision
        if 'precision' in self.history.history:
            axes[1, 0].plot(self.history.history['precision'], label='Train')
            axes[1, 0].plot(self.history.history['val_precision'], label='Validation')
            axes[1, 0].set_title('Precision')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Precision')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # Recall
        if 'recall' in self.history.history:
            axes[1, 1].plot(self.history.history['recall'], label='Train')
            axes[1, 1].plot(self.history.history['val_recall'], label='Validation')
            axes[1, 1].set_title('Recall')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Recall')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.model_dir / "training_history.png", dpi=300)
        plt.close()
        print(f"Training plots saved to {self.model_dir}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Train waste classification model")
    parser.add_argument("--data-dir", default="data/raw", help="Data directory")
    parser.add_argument("--model-dir", default="models", help="Model output directory")
    parser.add_argument("--epochs", type=int, default=20, help="Initial training epochs")
    parser.add_argument("--fine-tune-epochs", type=int, default=10, help="Fine-tuning epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--base-model", default="MobileNetV2", choices=["MobileNetV2", "EfficientNetB0"])
    parser.add_argument("--eval-only", action="store_true", help="Only evaluate existing model")
    args = parser.parse_args()
    
    classifier = WasteClassifier(
        data_dir=args.data_dir,
        model_dir=args.model_dir,
        batch_size=args.batch_size
    )
    
    if args.eval_only:
        # Load existing model
        model_path = Path(args.model_dir) / "best_model.keras"
        if model_path.exists():
            classifier.model = keras.models.load_model(model_path)
            print(f"Loaded model from {model_path}")
            classifier.evaluate()
        else:
            print(f"Model not found at {model_path}")
    else:
        # Build and train
        classifier.build_model(base_model_name=args.base_model)
        classifier.train(epochs=args.epochs, fine_tune_epochs=args.fine_tune_epochs)
        classifier.plot_training_history()
        classifier.evaluate()


if __name__ == "__main__":
    main()