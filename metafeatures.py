def expand_with_meta_features(df, text_col, stopwords):
    df = df.copy()
    stop_words = set(stopwords)
    
    df['words'] = df[text_col].str.split(' ').map(len)
    df['words_not_stopword'] = df[text_col].apply(
        lambda x: len([t for t in x.split(' ') if t not in stop_words]))
    df['commas'] = df[text_col].str.count(',')
    df['upper'] = df[text_col].map(lambda x: map(str.isupper, x)).map(sum)
    df['capitalized'] = df[text_col].map(lambda x: map(str.istitle, x)).map(sum)
    df['avg_word_length'] = df[text_col].apply(
        lambda x: np.mean([len(t) for t in x.split(' ') 
                           if t not in stop_words]) 
        if len([len(t) for t in x.split(' ') 
                if t not in stop_words]) > 0 else 0)
    
    return df