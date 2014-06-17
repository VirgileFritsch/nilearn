"""
This file provides the implementation of an object that can dynamically show
the progress of a procedure.

It is up to the Nilearn developers to adapt the ProgressBar object
within their procedures. They should identify milestones that define a
discretization of the procedure run (e.g. iterations for an algorithm,
downloaded file size for a downloader, number of processed subjects
for an operation on a multi-subject dataset). Then, the ProgressBar
object behaves like an iterator that generates and prints strings
on-demand.


"""
#Author: Aina Frau-Pascual, Jun. 2014
#        Virgile Fritsch, Jun. 2014, <virgile.fritsch@inria.fr>
import sys
import time
from multiprocessing.managers import BaseManager
import numpy as np


class ProgressBar:
    """Shows the progress of a procedure.

    The ProgressBar object tracks a procedure and is able to show its
    progress.  The object is instanciated with a predefined number of
    steps that the tracked procedure is supposed to perform
    (iterations typically). Then, the object behaves like an iterator:
    recurrent calls to the `show_progress` method are performed, each time
    evaluating the progress of the procedure and printing it.

    Parameters
    ----------
    n_steps : int,
      Number of steps the procedure is supposed to perform. More
      generally, steps correspond to milestones that the procedure is
      expected to reach (e.g. a number of iterations, a file size to
      be copied/downloaded, ...).  It is up to the caller to ensure
      that n_steps will actually be performed and --equivalently--
      that n_steps calls to the `show_progress` method will actually
      be performed.

    message : string format, len(message) <= 80
      Default message printed by the progress bar object. Attributes of the
      object can be referenced as named variables in format.
      In any case, the message is padded with spaces and fit into a
      80-character box.

    verbose : int, optional
      Define the verbosity level.
      0 corresponds to no display.


    Attributes
    ----------
    ``current_state_`` : float,
        Progress as a percentage. Starts at 0 %, ends at 100 %.

    ``step_size_`` : float,
        Progress "sampling rate". The precision with the which the progress
        of the procedure is evaluated.

    ``t0_`` : float,
        Starting time of the procedure calling the ProgressBar object.

    ``remaining_time_`` : float,
        Estimated remaning time before the procedure completes.
        It is useful to have it as an attribute to be able to print it
        using named variables in format strings.


    Caveats
    -------
    - A verbosity level of 0 does not print anything but potentially
      slows down the procedure the ProgressBar is tracking. Indeed, in
      a multiprocessing context, competting jobs still have to
      synchronize to update the object, even though nothing is
      actually printed.

    - In the multiprocessing case, competing jobs have to synchronize
      regarding the updates of the ProgressBar object. This can slow down
      the procedure a lot if a small discretization level (large `n_steps`)
      is provided by the user.

    - One problem with the use of n_steps is that it is up to the
      developer to ensure that the `show_progress` methods is called
      exactly `n_steps` times. He should be able to specify either a
      fixed number of steps or a starting point and a target point. In
      such a case, a supplemtnary argument would be passed to
      `show_progress` (e.g. `criterion_current_value`) in order to
      evaluate the current progress.


    """
    def __init__(self, n_steps, message=None, verbose=True):
        self.verbose = verbose
        if message is None:
            self.message = ("(%(current_state_)0.2f%% completed, "
                            "%(remaining_time_)d secs remaining)\r")
        self.current_state_ = 0.  # 0% accomplished at initialization time
        self.step_size_ = 100. / n_steps
        self.t0_ = time.time()
        self.remaining_time_ = np.iinfo(np.int).max
        if self.verbose:
            output_string = self.message % self.__dict__
            # constraint printing to 80 characters
            sys.stderr.write("%80s" % output_string)

    def get_verbose(self):
        """Return value of the `verbose` attribute.

        This method is compulsory to expose the `verbose` attribute in a
        concurrent access context (through the safeguarding `Manager` object)

        """
        return self.verbose

    def get_ellapsed_time(self):
        """Compute ellapsed time.
        """
        return time.time() - self.t0_

    def get_remaining_time(self):
        """Estimate remaining time based on ellapsed time.
        """
        estimated_total_time = (100. * self.get_ellapsed_time()
                                / self.current_state_)
        return estimated_total_time - self.get_ellapsed_time()

    def show_progress(self, additional_message=None):
        """Update and show progress.

        Parameters
        ----------
        additional_message: string, len(additional_message) <= 80
          Optional message to be displayed before the overall progress status
          is printed. It can be useful to give more details on what the
          tracked procedure is exactly doing.
          The additional message will be padded with spaces and fit into a
          80-character long, left-justified box.

        """
        # Update progress status
        self.current_state_ = self.current_state_ + self.step_size_
        self.remaining_time_ = self.get_remaining_time()

        # Print current status after update
        if self.verbose:
            # Optionally print a message that is context-specific and could
            # not be thought of in advance when the ProgressBar object is
            # created (e.g. printing the thread id when multiprocessing).
            if additional_message is not None:
                sys.stderr.write(additional_message.ljust(80))
            # Print general message, defined by the user or default.
            # This is where it may be useful to have the remaining
            # time as an explicit attribute.
            output_string = self.message % self.__dict__
            # constraint printing to 80 characters
            sys.stderr.write("%80s" % output_string)
            if self.current_state_ > (100 - self.step_size_):
                print ""


# Register the ProgressBar as a new type object that can be shared e.g. between
# processes (to safeguard against conflicting concurrent accesses).
class SharedProgressBar(BaseManager):
    """Object enclosing a ProgressBar for safe concurrent access.
    """
    pass

SharedProgressBar.register('Progress', ProgressBar)
