import datasets
import os
import dill
from pathlib import Path
from datasets import ClassLabel, DownloadConfig

logger = datasets.logging.get_logger(__name__)


_CITATION = ""
_DESCRIPTION = """\
"""


class SemRoleBankConfig(datasets.BuilderConfig):
    """BuilderConfig for SemRoleBankConfig"""
    
    def __init__(self, train_path:str, val_path:str, **kwargs):
        """BuilderConfig for SemRoleBankConfig.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        self.train_path = train_path
        self.val_path = val_path
        super(SemRoleBankConfig, self).__init__(**kwargs)

        
class SemRoleBank(datasets.GeneratorBasedBuilder):
    """Semanthai Bank dataset."""

    BUILDER_CONFIGS = [
        SemRoleBankConfig(name="std", train_path='../raw/train.data', val_path='../raw/val.data', version=datasets.Version("1.0.0"), description="Standard 80:20 Split"),
        SemRoleBankConfig(name="cv0", train_path='../raw/train_cv0.data', val_path='../raw/val_cv0.data', version=datasets.Version("1.0.0"), description="Cross-Validation Fold 0"),
        SemRoleBankConfig(name="cv1", train_path='../raw/train_cv1.data', val_path='../raw/val_cv1.data', version=datasets.Version("1.0.0"), description="Cross-Validation Fold 1"),
        SemRoleBankConfig(name="cv2", train_path='../raw/train_cv2.data', val_path='../raw/val_cv2.data', version=datasets.Version("1.0.0"), description="Cross-Validation Fold 2"),
        SemRoleBankConfig(name="cv3", train_path='../raw/train_cv3.data', val_path='../raw/val_cv3.data', version=datasets.Version("1.0.0"), description="Cross-Validation Fold 3"),
        SemRoleBankConfig(name="cv4", train_path='../raw/train_cv4.data', val_path='../raw/val_cv4.data', version=datasets.Version("1.0.0"), description="Cross-Validation Fold 4"),
        SemRoleBankConfig(name="oov", train_path='../raw/train_oov.data', val_path='../raw/val_cv3.data', version=datasets.Version("1.0.0"), description="Out-of-vocabulary split"),
    ]

    def __init__(self,
                 *args,
                 role_tags=('Accompanyment', 
                           'Agent', 
                           'Benefactor', 
                           'Experiencer', 
                           'Instrument', 
                           'Location', 
                           'Manner', 
                           'Measure', 
                           'Object', 
                           'Time', 
                           'Verb', 
                           'Z-O'),
                 **kwargs):
        self._role_tags = role_tags
        self._role_to_ix = dict((c, i) for i, c in enumerate(role_tags)) #convert ner to index
        # self._test_file = test_file
        super(SemRoleBank, self).__init__(*args, **kwargs)

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "tokens": datasets.Sequence(datasets.Value("string")),
                    "role_tags": datasets.Sequence(
                        datasets.features.ClassLabel(
                            names=sorted(list(self._role_tags))
                        )
                    )
                }
            ),
            supervised_keys=None,
            homepage="",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        
        print(self.config.version)
        urls_to_download = {
            "train": self.config.train_path,
            "val": self.config.val_path
        }
        downloaded_files = dl_manager.download_and_extract(urls_to_download)
        print(self.config)
        print(downloaded_files["train"])
        print(downloaded_files["val"])
        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepath": downloaded_files["train"]}),
            datasets.SplitGenerator(name=datasets.Split.VALIDATION, gen_kwargs={"filepath": downloaded_files["val"]}),
        ]

    def _generate_examples(self, filepath):
        logger.info("⏳ Generating examples from = %s", filepath)
        
        with open(filepath, 'rb') as file:
            sents = dill.load(file)
            
        for guid, sent in enumerate(sents):
            yield guid, {
                "id": str(guid),
                "tokens": ['_' if word[0] == '' else word[0] for word in sent],
                "role_tags": [self._role_to_ix[word[1]] for word in sent],
            }