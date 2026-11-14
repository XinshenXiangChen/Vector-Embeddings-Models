from collections import Counter
import re
from src.data.load_data import load_data

# GPT-2 style pre-tokenization regex optimized for contractions and punctuation
PRE_TOKEN_REGEX = re.compile(r"'s|'d|'m|'t|ll|ve|re|\w+|[^\w\s]", re.UNICODE)


class BPE:

  def __init__(self):
    self.dataset = load_data()
    self.vocab = {idx: bytes([idx]) for idx in range(256)}
    self.merges = {}
    self.word_dict = Counter()

  def tokenize(self):
    """Pass 1: Extract words via regex, convert to UTF-8 byte tuples,

    and count frequencies using a C-optimized Counter.
    """
    for dat_i in self.dataset["train"]:
      text = dat_i.get("text")

      if text is None:
        continue

      tokens = PRE_TOKEN_REGEX.findall(text)

      byte_tuples = (tuple(token.encode("utf-8")) for token in tokens)
      self.word_dict.update(byte_tuples)

    return self.word_dict

  def train(self, num_merges=1000):
    """Pass 2: Iteratively find and merge the most frequent adjacent pairs."""
    if not self.word_dict:
      self.tokenize()

    current_word_dict = dict(self.word_dict)

    for i in range(num_merges):
      # 1. Count all adjacent pairs across all unique words, weighted by frequency
      pair_counts = Counter()
      for word_tuple, freq in current_word_dict.items():
        for pair in zip(word_tuple, word_tuple[1:]):
          pair_counts[pair] += freq

      if not pair_counts:
        print("No more pairs left to merge.")
        break

      # 2. Find the most frequent pair
      best_pair = max(pair_counts, key=pair_counts.get)

      # 3. Mint a new token ID (starting at 256 and incrementing)
      new_id = 256 + i

      # 4. Record the merge and update the vocabulary
      self.merges[best_pair] = new_id
      self.vocab[new_id] = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]

      # 5. Apply the merge across all unique words in the working dictionary
      current_word_dict = self._apply_merge_to_dict(
          current_word_dict, best_pair, new_id
      )

      # Optional status update every 50 merges
      if (i + 1) % 50 == 0 or i == num_merges - 1:
        print(
            f"Merge {i+1}/{num_merges}: Pair {best_pair} -> New Token ID"
            f" {new_id} (Freq: {pair_counts[best_pair]})"
        )

    return self.merges, self.vocab

  def _apply_merge_to_dict(self, word_dict, pair, new_id):
    """Helper function to replace a target pair with the new_id

    in all unique word tuples.
    """
    new_word_dict = {}
    p0, p1 = pair
    for word_tuple, freq in word_dict.items():
      new_tokens = []
      idx = 0
      while idx < len(word_tuple):
        # If we find the target pair, replace it with new_id and skip the next element
        if (
            idx < len(word_tuple) - 1
            and word_tuple[idx] == p0
            and word_tuple[idx + 1] == p1
        ):
          new_tokens.append(new_id)
          idx += 2
        else:
          new_tokens.append(word_tuple[idx])
          idx += 1
      new_word_dict[tuple(new_tokens)] = freq
    return new_word_dict


if __name__ == "__main__":
  _BPE = BPE()

  print("Step 1: Running first-pass tokenization and frequency counting...")
  _BPE.tokenize()
  print(f"Found {len(_BPE.word_dict)} unique base words/tokens.")

  print("\nStep 2: Training BPE (running merge iterations)...")
  # Change num_merges to however large you want your vocabulary to grow (e.g., 500, 1000, 5000)
  merges, vocab = _BPE.train(num_merges=200)

  print(f"\nTraining complete!")
  print(f"Final Vocabulary Size: {len(vocab)}")
  print(f"Total Merges Learned: {len(merges)}")