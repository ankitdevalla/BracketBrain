"""
Script to inspect the feature importance of the model with tempo features.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

# Set the style for plots
plt.style.use('fivethirtyeight')
sns.set_palette("deep")

def main():
    print("Inspecting model with tempo features...")
    
    # Check if model exists
    model_path = 'scripts/final_model_with_tempo.pkl'
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return
    
    # Load the model
    try:
        model = joblib.load(model_path)
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return
    
    # Load feature names
    try:
        feature_names = np.load('scripts/feature_names_with_tempo.npy', allow_pickle=True)
        print(f"Loaded {len(feature_names)} features")
    except Exception as e:
        print(f"Error loading feature names: {str(e)}")
        feature_names = None
    
    # Extract feature importances
    if hasattr(model, 'named_steps') and 'classifier' in model.named_steps:
        classifier = model.named_steps['classifier']
        
        if hasattr(classifier, 'feature_importances_'):
            importances = classifier.feature_importances_
            
            if feature_names is not None:
                # Create DataFrame with feature names and importances
                feature_importance = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': importances
                }).sort_values('Importance', ascending=False)
                
                print("\nTop 10 Feature Importances:")
                print(feature_importance.head(10))
                
                # Check rank of tempo-related features
                tempo_features = feature_importance[
                    feature_importance['Feature'].str.contains('Tempo|Poss')
                ]
                
                print("\nTempo-related Feature Importances:")
                print(tempo_features)
                
                # Plot feature importances
                plt.figure(figsize=(12, 8))
                sns.barplot(x='Importance', y='Feature', data=feature_importance.head(15))
                plt.title('Top 15 Feature Importance')
                plt.tight_layout()
                plt.savefig('output/feature_importance_with_tempo.png')
                print("Feature importance plot saved to 'output/feature_importance_with_tempo.png'")
                
                # Plot tempo features specifically
                if not tempo_features.empty:
                    plt.figure(figsize=(10, 6))
                    sns.barplot(x='Importance', y='Feature', data=tempo_features)
                    plt.title('Tempo-related Feature Importance')
                    plt.tight_layout()
                    plt.savefig('output/tempo_feature_importance.png')
                    print("Tempo feature importance plot saved to 'output/tempo_feature_importance.png'")
            else:
                print("\nFeature Importances (without names):")
                for i, importance in enumerate(importances):
                    print(f"Feature {i}: {importance}")
        else:
            print("Model doesn't have feature_importances_ attribute")
    else:
        print("Model doesn't have the expected structure")

if __name__ == "__main__":
    main()
