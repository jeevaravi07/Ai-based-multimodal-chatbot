"""
One-command build. Run this once (and again any time data/cases.csv,
data/images/, or data/symptom_checker/ change):

    python scripts/build_index.py

Steps:
    1. Parse case_text into CARE sections + auto-sort images by scan type
    2. Fit the TF-IDF+SVD text encoder on all case texts, build the text FAISS index
    3. Encode every case image (handcrafted descriptor), train the modality
       classifier, build the image FAISS index
    4. Train the symptom checker (independent pipeline)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from rag import prepare_data, text_embeddings, image_retriever, vector_store, symptom_checker


def main():
    print("=" * 60)
    print("Step 1/4: Parsing case text into CARE sections + sorting images")
    print("=" * 60)
    records = prepare_data.build_dataset()
    print(f"  {len(records)} rows parsed.")

    print()
    print("=" * 60)
    print("Step 2/4: Fitting text encoder (TF-IDF+SVD) + building text FAISS index")
    print("=" * 60)
    texts = [r["case_text"] for r in records]
    text_vectors = text_embeddings.fit_text_encoder(texts)
    text_index = vector_store.build_index(text_vectors)
    vector_store.save_index(text_index, config.TEXT_FAISS_INDEX)
    print(f"  Text index built: {text_index.ntotal} vectors, dim={text_vectors.shape[1]}")

    print()
    print("=" * 60)
    print("Step 3/4: Encoding images, training modality classifier, building image FAISS index")
    print("=" * 60)
    image_paths = [r["image_path"] for r in records]
    modalities = [r["modality"] for r in records]
    image_retriever.train_modality_classifier(image_paths, modalities)
    image_vectors = [image_retriever.encode_image(p) for p in image_paths]
    import numpy as np
    image_vectors = np.stack(image_vectors)
    image_index = vector_store.build_index(image_vectors)
    vector_store.save_index(image_index, config.IMAGE_FAISS_INDEX)
    print(f"  Image index built: {image_index.ntotal} vectors, dim={image_vectors.shape[1]}")

    vector_store.save_metadata(records)
    print(f"  Metadata saved for {len(records)} records.")

    print()
    print("=" * 60)
    print("Step 4/4: Training symptom checker (independent pipeline)")
    print("=" * 60)
    symptom_checker.train(verbose=True)

    print()
    print("All done. Start the app with: python app.py")


if __name__ == "__main__":
    main()
