
class TMonitor(Thread):
    def run(self):
        cur_t = self._time()
        while True:
            # After processing and before sleeping, notify that we woke
            # Need to be done just before sleeping
            self.woken = cur_t
            # Sleep some time...
            self.was_killed.wait(self.sleep_interval)
            # Quit if killed
            if self.was_killed.is_set():
                return
            # Then monitor!
            # Acquire lock (to access _instances)
            with self.tqdm_cls.get_lock():
                cur_t = self._time()
                # Check tqdm instances are waiting too long to print
                instances = self.get_instances()
                for instance in instances:
                    # Check event in loop to reduce blocking time on exit
                    if self.was_killed.is_set():
                        return
                    # Only if mininterval > 1 (else iterations are just slow)
                    # and last refresh exceeded maxinterval
                    if instance.miniters > 1 and \
                            (cur_t - instance.last_print_t) >= \
                            instance.maxinterval:
                        # force bypassing miniters on next iteration
                        # (dynamic_miniters adjusts mininterval automatically)
                        instance.miniters = 1
                        # Refresh now! (works only for manual tqdm)
                        instance.refresh(nolock=True)
                if instances != self.get_instances():  # pragma: nocover
                    warn("Set changed size during iteration" +
                         " (see https://github.com/tqdm/tqdm/issues/481)",
                         TqdmSynchronisationWarning, stacklevel=2)
