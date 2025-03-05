import pickle
import numpy as np

def inspect_model(model_path='scripts/final_model.pkl'):
    """Inspect the contents of the model file."""
    try:
        with open(model_path, 'rb') as f:
            raw_data = f.read()
            print(f"\nRead {len(raw_data)} bytes from {model_path}")
            
            # Try multiple unpickle operations
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
                        elif hasattr(obj, 'get_params'):
                            print("ML Model details:")
                            print(f"Parameters: {obj.get_params()}")
                            if hasattr(obj, 'coef_'):
                                print(f"Coefficients shape: {obj.coef_.shape}")
                                print(f"First few coefficients: {obj.coef_[:5]}")
                            if hasattr(obj, 'intercept_'):
                                print(f"Intercept: {obj.intercept_}")
                        else:
                            print(f"Object contents: {obj}")
                            
                    except EOFError:
                        break  # Reached end of file
                    except Exception as e:
                        print(f"Error reading next object: {e}")
                        break
                        
            except Exception as e:
                print(f"\nError during unpickling: {e}")
                
                # Try to interpret as numpy array with different dtypes
                for dtype in [np.float64, np.float32, np.int64, np.int32]:
                    try:
                        array_data = np.frombuffer(raw_data, dtype=dtype)
                        print(f"\nInterpreted as numpy array with dtype {dtype}:")
                        print(f"Shape: {array_data.shape}")
                        print(f"First few values: {array_data[:5]}")
                    except Exception:
                        continue
                    
    except Exception as e:
        print(f"Error reading model file: {e}")
        return None

if __name__ == "__main__":
    model = inspect_model()
