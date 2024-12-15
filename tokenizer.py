import json

import regex as re


def bytes_to_unicode():
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )  # list of unicode bytes ignoring some control characters which cause problem in BPE
    cs = bs[:]  # repica of bytes to store the characters representing that bytes
    n = 0
    for b in range(2**8):  # iterate through all the possible bytes
        if b not in bs:  # if we have skipped a particular byte
            bs.append(b)  # append it to the orig list
            cs.append( 2**8 + n)  # append a substitute unicode character for the one we have skipped
            n += 1
    cs = [ chr(n) for n in cs ]  # convert the unicode bytes to the corresponding characters
    return dict(zip(bs, cs))  # create a mapping of bytes -> characters


def get_pairs(word):
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs  # hello -> { ('h', 'e'), ('e', 'l'), ('l', 'l'), ('l', 'o') }


class Tokenizer:
    def __init__(
        self,
        vocab_file,
        merges_file,
        special_tokens_file,
        errors="replace",
        unk_token="<|endoftext|>",
        bos_token="<|im_start|>",
        eos_token="<|im_end|>",
        pad_token="<|im_end|>",
    ):
        with open(vocab_file, encoding="utf-8") as vocab_handle:
            self.encoder = json.load( vocab_handle)  # this is mapping of string to its corresponding id
        self.decoder = { v: k for k, v in self.encoder.items() }  # inverse mapping from id to string
        self.errors = errors  # how to handle errors in decoding in utf-8
        self.byte_encoder = bytes_to_unicode()  # mapping of bytes to unicode chars as explained above
        self.byte_decoder = { v: k for k, v in self.byte_encoder.items() }  # inverse mapping of byte_decoder
        with open(merges_file, encoding="utf-8") as merges_handle:
            bpe_merges = merges_handle.read().split("\n")[1:-1]
        bpe_merges = [ tuple(merge.split()) for merge in bpe_merges ]  # this creates tuples of merges from the txt file which has merge on each line seperated by a space
        self.bpe_ranks = dict( zip(bpe_merges, range(len(bpe_merges))))  # this stores the order in which merges happened while training BPE as it is used while inference (mapping of merge_tuple -> merge_rank)
        self.cache = {}  # cache to speed up inference
        if isinstance(special_tokens_file, str):
            with open(special_tokens_file, encoding="utf-8") as f:
                self.special = json.load( f)  # this is mapping of special tokens to their corresponding id (eg. {"<|im_end|>":1})
        elif isinstance(special_tokens_file, dict):
            self.special = special_tokens_file
        self.pat = re.compile( r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")  # this is the pattern used by GPT-2 to split up the text before BPE
        self.unk_token = unk_token  # storing some special tokens as attributes
        self.eos_token = eos_token
        self.bos_token = bos_token
        self.pad_token = pad_token

    def bpe(self, token: str):
        if token in self.cache:
            return self.cache[token]  # if in cache return directly
        word = tuple(token)
        pairs = get_pairs(
            word
        )  # hello -> { ('h', 'e'), ('e', 'l'), ('l', 'l'), ('l', 'o') }

        if not pairs:
            return token  # return as it is for single token (eg. h -> h)

        while True:
            bigram = min( pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf")))  # find the pair to merge which has the least rank from the merge ranks mapping
            if bigram not in self.bpe_ranks:
                break  # if not found that means there is nothin to merge so break the while
            first, second = bigram
            new_word = []
            i = 0
            while i < len( word):  # this while loop will do the merging if the identified bigrams
                try:
                    j = word.index(first, i)
                except ValueError:
                    new_word.extend(word[i:])
                    break
                else:
                    new_word.extend(word[i:j])
                    i = j

                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if ( len(word) == 1):  # break if everything is merged into a single word as we need atleast two things to merge
                break
            else:
                pairs = get_pairs(word)  # calculate the pairs again after the bpe merges
        word = " ".join(word)  # here we combine the bpe ouptput to single text with " " seperator
        self.cache[token] = word  # store the token in cache
        return word

    def _tokenize(self, text):
        """Tokenize a string."""
        bpe_tokens = []
        for text_chunk in re.findall(self.pat, text):
            utf_8_encoded_text_chunk = text_chunk.encode()
            byte_encoded_text_chunk = [
                self.byte_encoder[b] for b in utf_8_encoded_text_chunk
            ]
            concatenated_byte_encoded_text_chunk = "".join(byte_encoded_text_chunk)
            chunk_bpe_tokens = self.bpe(concatenated_byte_encoded_text_chunk).split(" ")
            bpe_tokens.extend(chunk_bpe_tokens)
        # print(f"_tokenize({text})={bpe_tokens}")
        return bpe_tokens

    def tokenize(self, text):
        import regex as re

        special_pattern = ( "(" + "|".join(re.escape(k) for k in self.special) + ")")  # regex pattern to select special tokens
        text_chunks = re.split( special_pattern, text)  # splits the text around special tokens
        tokenized_text = []  # Store the tokenized text output
        for chunk in text_chunks:
            if not chunk:  # ignore the empty strings if any after the split
                continue
            if chunk in self.special:
                tokenized_text.append(chunk)  # if special token then append it as it is
            else:
                tokenized_text.extend(
                    self._tokenize(chunk)
                )  # if not special apply bpe to the chunk
        # print(f"tokenize({text})={tokenized_text}")
        return tokenized_text

    def convert_tokens_to_ids(self, tokens):
        ids = []
        for token in tokens:
            if token in self.special:
                ids.append(self.special[token])
            else:
                id_ = self.encoder.get(token, self.encoder.get(self.unk_token))
                ids.append(id_)
        # print(f"convert_tokens_to_ids({tokens}) = {ids}")
        return ids

    def encode(self, text:str | None = None, conversation:list[dict] | None = None, add_generation_prompt:bool = False):
        end_of_turn_text = "<|im_end|>\n"
        system_start_text = "<|im_start|>system\n"
        user_start_text = "<|im_start|>user\n"
        assistant_start_text = "<|im_start|>assistant\n" 
        if conversation is not None:
            text = ""
            for item in conversation:
                assert item.keys() == {"role", "content"}, "conversation key must be one of role, content"
                role = item["role"]
                content = item["content"]
                if role == "system":
                    text += system_start_text + content + end_of_turn_text
                elif role == "user":
                    text += user_start_text + content + end_of_turn_text
                elif role == "assistant":
                    text += assistant_start_text + content + end_of_turn_text
            if add_generation_prompt:
                text += assistant_start_text 
            return self.convert_tokens_to_ids(self.tokenize(text))
        elif text is not None:
                return self.convert_tokens_to_ids(self.tokenize(text))
        else:
            raise ValueError("Either text or conversation must be provided")
            

    def convert_ids_to_tokens(self, ids, skip_special_tokens: bool = False):
        tokens = []
        for index in ids:
            index = int(index)
            if skip_special_tokens and index in self.special:
                continue
            else:
                token = self.decoder.get(index)
                tokens.append(token)
        return tokens

    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (string) in a single string."""
        text = "".join(tokens)
        # print(f"join({tokens}) = {text}")
        byte_decoded_text = [self.byte_decoder[c] for c in text]
        # print(f"byte_decoder({text}) = {byte_decoded_text}")
        text_ = bytearray(byte_decoded_text).decode("utf-8", errors=self.errors)
        # print(f"decode_utf_8((byte_decoder({text})) = {text_}")
        return text_

    def decode(
        self,
        token_ids,
        skip_special_tokens: bool = False,
    ) -> str:
        filtered_tokens = self.convert_ids_to_tokens(
            token_ids, skip_special_tokens=skip_special_tokens
        )
        # print(f"convert_ids_to_tokens({token_ids}) = {filtered_tokens}")
        sub_texts = []
        current_sub_text = []
        for token in filtered_tokens:
            if skip_special_tokens and token in self.special:
                continue
            else:
                current_sub_text.append(token)
        if current_sub_text:
            sub_texts.append(self.convert_tokens_to_string(current_sub_text))
        # print(f"convert_tokens_to_string({filtered_tokens}) = {sub_texts}")
        text = "".join(sub_texts)
        # print(f"join({sub_texts}) = {text}")
        return text
