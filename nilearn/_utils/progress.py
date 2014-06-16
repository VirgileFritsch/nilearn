import sys
import time



class progress_bar(object):
    """
    Shows the progress of the process
    
    Parameters:
    ----------
    - n_features: number of features
    - n_voxels: number of voxels
    - thread_id: process id
    - verbose: level of verbosity
    """
    
    def __init__(self, n_features = 1, n_voxels = 1,
                       thread_id = 0, verbose = 0):
        self.n_features = n_features
        self.n_voxels = n_voxels
        self.thread_id = thread_id
        self.verbose = verbose
        # step (depending on verbose) 
        self.step = 11 - min(verbose, 10)
        # starting timepoint: declare object when t0
        self.t0 = time.time()
        
        
    def measure_progress(self, ifeat, message):
        """ 
        Measures progress: done and remaining.
        
        Parameters:
        ----------
        - ifeat: feature being processed
        - message: additional message
        """
        if (ifeat % self.step == 0):
            # If there is only one job, progress information is fixed
            if self.n_voxels == self.n_features:
                crlf = "\r"
            else:
                crlf = "\n"
            # Compute percentage
            percent = float(ifeat) / self.n_features
            percent = round(percent * 100, 2)
            # Compute time
            dt = time.time() - self.t0
            # Compute remaining
            #    We use a max to avoid a division by zero
            remaining = (100. - percent) / max(0.01, percent) * dt
            # Print results
            sys.stderr.write(
                "Job #%d, processed %d/%d voxels "
                "(%0.2f%%, %i secs remaining). %s %s"
                % (self.thread_id, ifeat, self.n_features, percent, remaining, message, crlf))
            
            
            