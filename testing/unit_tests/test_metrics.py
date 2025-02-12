from src.metrics import custom_r2
import joblib
import numpy as np



class TestMetrics:
    """
    Handles all the test cases related to metrics, remember to prefix function names with ``test_``  to let ``pytest`` discover them.  
    """
    
    def test_r2_score(self):
        
        """
        The function generates a test for the ``custom_r2`` score function, it tests it using a synthetically generated data.
        """
        
        #Read the dummy data 
        with open("data/test_data/dummy_test_data_r2.joblib","rb") as f:
            r2_data = joblib.load(f)
        
        y_true, y_pred, mask =r2_data["true"], r2_data["predictions"], r2_data["mask"]

        #Calculate r2_score on dummy data         
        r2_score = custom_r2(y_true=y_true,
                         y_pred=y_pred,
                        mask=mask)
        #Check assertion 
        assert  abs(np.round(r2_score,2)) == 1.06
        
        
        