import os
import torch
import random
import warnings
import numpy as np
from argparse import ArgumentParser

from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
from transformers import BertTokenizer, BertConfig, BertForMaskedLM
# from transformers.trainer_utils import get_last_checkpoint
from transformers import TextDataset
from transformers.trainer_utils import get_last_checkpoint


def pre_train(args, model):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = BertTokenizer.from_pretrained(args.vocab_path, model_max_length=args.seq_length)
    bert_config = BertConfig.from_pretrained(args.config_path)
    model = BertForMaskedLM(config=bert_config)
    model = model.to(device)

    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=True, mlm_probability=0.15)

    training_args = TrainingArguments(
        seed=args.seed,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        output_dir=args.checkpoint_save_path,
        learning_rate=args.learning_rate,
        save_total_limit=args.save_total_limit,
        # num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.batch_size,
        report_to="tensorboard",
        do_train=True,
        do_eval=False,
        num_train_epochs=1,
    )

    print("=========loading TextDateset=========")
    dataset = TextDataset(tokenizer=tokenizer, block_size=args.seq_length, file_path=args.pretrain_data_path)
    print("=========TextDateset loaded =========")

    trainer = Trainer(model, args=training_args, train_dataset=dataset, data_collator=data_collator)

    last_checkpoint = get_last_checkpoint(training_args.output_dir)
    if last_checkpoint is not None:
        train_result = trainer.train(resume_from_checkpoint=last_checkpoint)
    else:
        print("=========training=========")
        train_result = trainer.train()
    print(train_result)
    trainer.save_model(args.pretrain_save_path)
    tokenizer.save_vocabulary(args.pretrain_save_path)

    return model.bert



