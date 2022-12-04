from simpleclassifier import target_classifier
from perspective_client import perspective_client
from CounterSpeechGenerator import GPT2Generator


if __name__ == '__main__':
    text = 'you are a terrible gay.'
    # init_target_classifier = target_classifier()

    # text_list = []
    # text_list.append(text)
    # target, prob = init_target_classifier.classify(text_list)
    # print(f'{target}:{prob}')
    #
    # init_GPT2_generator = GPT2Generator()
    # response = init_GPT2_generator.generate_counterspeech(text, target)
    # print(response)


    init_perspective_client = perspective_client()
    analyse_result = init_perspective_client.get_toxicity(text)
    print(analyse_result)
