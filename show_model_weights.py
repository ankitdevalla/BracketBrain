"""
Script to extract and display the weights for each variable in the model.
"""

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree

def main():
    print("Extracting model weights and feature importance...")
    
    # Load the model
    model = joblib.load('scripts/final_model_with_tempo.pkl')
    
    # Load feature names
    feature_names = np.load('scripts/feature_names_with_tempo.npy', allow_pickle=True)
    
    # Get the classifier from the pipeline
    if hasattr(model, 'named_steps') and 'classifier' in model.named_steps:
        classifier = model.named_steps['classifier']
        
        # Get feature importances
        if hasattr(classifier, 'feature_importances_'):
            importances = classifier.feature_importances_
            
            # Create DataFrame with feature names and importances
            feature_importance = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importances
            }).sort_values('Importance', ascending=False)
            
            print("\nFeature Importances (weights):")
            for i, row in feature_importance.iterrows():
                print(f"{row['Feature']}: {row['Importance']:.6f}")
            
            # Plot feature importances
            plt.figure(figsize=(12, 10))
            sns.barplot(x='Importance', y='Feature', data=feature_importance)
            plt.title('Feature Importance (Model Weights)')
            plt.tight_layout()
            plt.savefig('output/model_weights.png')
            print("\nModel weights plot saved to 'output/model_weights.png'")
            
            # If it's a Random Forest, we can look at the first few trees
            if isinstance(classifier, RandomForestClassifier):
                print("\nRandom Forest Classifier with", len(classifier.estimators_), "trees")
                
                # Visualize the first decision tree in the forest
                plt.figure(figsize=(20, 10))
                plot_tree(classifier.estimators_[0], 
                          feature_names=feature_names,
                          filled=True, 
                          max_depth=3,
                          fontsize=10)
                plt.title('First Decision Tree in the Random Forest (Limited to Depth 3)')
                plt.tight_layout()
                plt.savefig('output/decision_tree_example.png')
                print("Example decision tree saved to 'output/decision_tree_example.png'")
                
                # Show the top features used in the first split of each tree
                print("\nTop features used in the first split of each tree:")
                first_split_features = []
                for tree in classifier.estimators_[:20]:  # Look at first 20 trees
                    if tree.tree_.feature[0] >= 0:  # If it's not a leaf node
                        first_split_features.append(feature_names[tree.tree_.feature[0]])
                
                # Count occurrences of each feature
                feature_counts = pd.Series(first_split_features).value_counts()
                print(feature_counts)
        else:
            print("Model doesn't have feature_importances_ attribute")
    else:
        print("Model doesn't have the expected structure")

if __name__ == "__main__":
    main()
