import logging
import os

# os.environ["CUDA_VISIBLE_DEVICES"] = ""

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.WARN,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
