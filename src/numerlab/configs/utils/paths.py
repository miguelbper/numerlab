import os

from rootutils import find_root

root_dir = str(find_root(search_from=__file__))
data_dir = os.path.join(root_dir, "data")
log_dir = os.path.join(root_dir, "logs")
output_dir = "${hydra:runtime.output_dir}"
work_dir = "${hydra:runtime.cwd}"
