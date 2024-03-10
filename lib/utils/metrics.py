def print_status_bar(iteration, total, loss, metrics=None, bar_size=45):
    '''This function prints out the loss + other metrics (if specified) one below another.
    iteration -> Epoch
    total     -> Total Epochs
    loss      -> Model loss for that epoch
    metrics   -> Calculated metrics'''

    metrics = ' - '.join(["{}: {:.4f}".format(m.name, m.result())
                         for m in [loss] + (metrics or [])])
    end = "" if iteration < total else "\n"

    # \r ensures that the output is printed in the same line
    # print ("\r{}/{} - ".format(iteration, total) + metrics, end=end)

    # \r ensures that the output is printed in the same line
    # Change bar_size to get larger or smaller bars
    print("\r[{}{}] - "
          .format(("=" * int(iteration/total*bar_size) + ">"),
                  ("." * int((total-iteration)/total*bar_size))
                  ) + metrics, end=end)
