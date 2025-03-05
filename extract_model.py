import pickle
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib

def extract_model(model_path='scripts/final_model.pkl'):
    """Extract and analyze the model from the pickle file."""
    try:
        print(f"\nTrying to load model with joblib...")
        try:
            model = joblib.load(model_path)
            print(f"\nSuccessfully loaded model with joblib!")
            print(f"Model type: {type(model)}")
            
            if isinstance(model, Pipeline):
                print("\nPipeline details:")
                print(f"Steps: {[step[0] for step in model.steps]}")
                
                # Extract and save model coefficients/feature importances
                for name, step in model.named_steps.items():
                    print(f"\nAnalyzing step: {name}")
                    print(f"Type: {type(step)}")
                    
                    if isinstance(step, StandardScaler):
                        print("Found StandardScaler:")
                        print(f"Mean: {step.mean_}")
                        print(f"Scale: {step.scale_}")
                        np.save('model/scaler_mean.npy', step.mean_)
                        np.save('model/scaler_scale.npy', step.scale_)
                        print("\nSaved scaler parameters")
                        
                    elif isinstance(step, RandomForestClassifier):
                        print("Found RandomForestClassifier:")
                        print(f"N estimators: {step.n_estimators}")
                        print(f"Feature importances: {step.feature_importances_}")
                        np.save('model/feature_importances.npy', step.feature_importances_)
                        print("\nSaved feature importances")
                        
                        # Save the entire model step
                        with open('model/classifier.pkl', 'wb') as model_file:
                            pickle.dump(step, model_file)
                        print("\nSaved classifier model")
                        
            elif isinstance(model, RandomForestClassifier):
                print("\nFound RandomForestClassifier:")
                print(f"N estimators: {model.n_estimators}")
                print(f"Feature importances: {model.feature_importances_}")
                np.save('model/feature_importances.npy', model.feature_importances_)
                print("\nSaved feature importances")
                
                # Save the entire model
                with open('model/classifier.pkl', 'wb') as model_file:
                    pickle.dump(model, model_file)
                print("\nSaved classifier model")
                
        except Exception as e:
            print(f"\nError loading with joblib: {e}")
            print("\nTrying standard pickle loading...")
            
            with open(model_path, 'rb') as f:
                raw_data = f.read()
                print(f"\nRead {len(raw_data)} bytes from {model_path}")
                
                try:
                    f.seek(0)
                    while True:
                        try:
                            obj = pickle.load(f)
                            print("\nFound pickled object:")
                            print(f"Type: {type(obj)}")
                            
                            if isinstance(obj, np.ndarray):
                                print("Array details:")
                                print(f"Shape: {obj.shape}")
                                print(f"Dtype: {obj.dtype}")
                                print(f"Values: {obj}")
                                
                                # Save feature names
                                if obj.dtype == np.dtype('O'):  # Object dtype indicates strings
                                    np.save('model/feature_names.npy', obj)
                                    print("\nSaved feature names to model/feature_names.npy")
                            else:
                                print(f"Object contents: {obj}")
                                
                        except EOFError:
                            break  # Reached end of file
                        except Exception as e:
                            print(f"Error reading next object: {e}")
                            break
                            
                except Exception as e:
                    print(f"\nError during unpickling: {e}")
                    
    except Exception as e:
        print(f"Error reading model file: {e}")
        return None

if __name__ == "__main__":
    extract_model()
