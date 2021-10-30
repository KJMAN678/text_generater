import streamlit as st
from transformers import T5Tokenizer, AutoModelForCausalLM

def cached_tokenizer():
    tokenizer = T5Tokenizer.from_pretrained("rinna/japanese-gpt2-medium")
    tokenizer.do_lower_case = True
    return tokenizer

def cached_model():
    model = AutoModelForCausalLM.from_pretrained("rinna/japanese-gpt2-medium")
    return model

def main():
  st.title("GPT-2による日本語の文章生成")

  num_of_output_text = st.slider(label='出力する文章の数',
                  min_value=1,
                  max_value=2,
                  value=1,
                  )

  length_of_output_text = st.slider(label='出力する文字数',
                  min_value=30,
                  max_value=200,
                  value=100,
                  )

  PREFIX_TEXT = st.text_area(
        label='テキスト入力', 
        value='吾輩は猫である'
  )

  progress_num = 0
  status_text = st.empty()
  progress_bar = st.progress(progress_num)

  if st.button('文章生成'):

    st.text("読み込みに時間がかかります")
    progress_num = 10
    status_text.text(f'Progress: {progress_num}%')
    progress_bar.progress(progress_num)

    tokenizer = cached_tokenizer()
    progress_num = 25
    status_text.text(f'Progress: {progress_num}%')
    progress_bar.progress(progress_num)

    model = cached_model()
    progress_num = 40
    status_text.text(f'Progress: {progress_num}%')
    progress_bar.progress(progress_num)

    # 推論 
    input = tokenizer.encode(PREFIX_TEXT, return_tensors="pt") 
    progress_num = 60
    status_text.text(f'Progress: {progress_num}%')
    progress_bar.progress(progress_num)

    output = model.generate(
            input, do_sample=True, 
            max_length=length_of_output_text,
            num_return_sequences=num_of_output_text
            )
    progress_num = 90
    status_text.text(f'Progress: {progress_num}%')
    progress_bar.progress(progress_num)

    output_text = "".join(tokenizer.batch_decode(output)).replace("</s>", "")
    progress_num = 95
    status_text.text(f'Progress: {progress_num}%')
    progress_bar.progress(progress_num)

    st.info('生成結果')
    progress_num = 100
    status_text.text(f'Progress: {progress_num}%')
    st.write(output_text)
    progress_bar.progress(progress_num)

if __name__ == '__main__':
  main()