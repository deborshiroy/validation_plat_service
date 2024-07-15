import torch
from sacrebleu import corpus_bleu
from rouge_score import rouge_scorer
from bert_score import score
from transformers import GPT2LMHeadModel, GPT2Tokenizer, pipeline
import nltk
from nltk.util import ngrams

class Evaluator:
    def __init__(self):
        try:
            self.gpt2_model = GPT2LMHeadModel.from_pretrained('gpt2')
            self.gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        except Exception as e:
            print(traceback.format_exc())

    def evaluate_bleu_rouge(self, candidates, references):
        try:
            bleu_score = corpus_bleu(candidates, [references]).score
            scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
            rouge_scores = [scorer.score(ref, cand) for ref, cand in zip(references, candidates)]
            rouge1 = sum([score['rouge1'].fmeasure for score in rouge_scores]) / len(rouge_scores)
            rouge2 = sum([score['rouge2'].fmeasure for score in rouge_scores]) / len(rouge_scores)
            rougeL = sum([score['rougeL'].fmeasure for score in rouge_scores]) / len(rouge_scores)
            return bleu_score, rouge1, rouge2, rougeL
        except Exception as e:
            print(traceback.format_exc())

    def evaluate_bert_score(self, candidates, references):
        try:
            P, R, F1 = score(candidates, references, lang="en", model_type='bert-base-multilingual-cased')
            return P.mean().item(), R.mean().item(), F1.mean().item()
        except Exception as e:
            print(traceback.format_exc())

    def evaluate_perplexity(self, text):
        try:
            encodings = self.gpt2_tokenizer(text, return_tensors='pt')
            max_length = self.gpt2_model.config.n_positions
            stride = 512
            lls = []
            for i in range(0, encodings.input_ids.size(1), stride):
                begin_loc = max(i + stride - max_length, 0)
                end_loc = min(i + stride, encodings.input_ids.size(1))
                trg_len = end_loc - i
                input_ids = encodings.input_ids[:, begin_loc:end_loc]
                target_ids = input_ids.clone()
                target_ids[:, :-trg_len] = -100
                with torch.no_grad():
                    outputs = self.gpt2_model(input_ids, labels=target_ids)
                    log_likelihood = outputs[0] * trg_len
                lls.append(log_likelihood)
            ppl = torch.exp(torch.stack(lls).sum() / end_loc)
            return ppl.item()
        except Exception as e:
            print(traceback.format_exc())


    def evaluate_all(self, question, response, reference):
        try:
            candidates = [response]
            references = [reference]
            bleu, rouge1, rouge2, rougeL = self.evaluate_bleu_rouge(candidates, references)
            bert_p, bert_r, bert_f1 = self.evaluate_bert_score(candidates, references)
            perplexity = self.evaluate_perplexity(response)

            return {
                "BLEU": bleu,
                "ROUGE-1": rouge1,
                "ROUGE-2": rouge2,
                "ROUGE-L": rougeL,
                "BERT P": bert_p,
                "BERT R": bert_r,
                "BERT F1": bert_f1,
                "Perplexity": perplexity

            }
        except Exception as e:
            print(traceback.format_exc())