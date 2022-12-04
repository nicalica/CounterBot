from typing import Dict
from typing import NamedTuple
import csv
import re


import googleapiclient
from googleapiclient import discovery


PERSPECTIVE_API_KEY = 'PERSPECTIVE_API_KEY'
TWITTER_BEARER_TOKEN = 'TWITTER_BEARER_TOKEN'


class ToxicTweet(NamedTuple):
    tweet: Dict[str, object]
    analysis: Dict[str, object]


class perspective_client:

    def __init__(self, keys_file='./data/.api_keys'):
        self.get_api_keys(keys_file)
        self.service = discovery.build(
            'commentanalyzer', 'v1alpha1', developerKey=self.api_keys[PERSPECTIVE_API_KEY],
            discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1",
        )
        self.attributes = ['SEVERE_TOXICITY', 'IDENTITY_ATTACK', 'SEXUALLY_EXPLICIT']

    def get_api_keys(self, keys_file):
        self.api_keys = {}
        with open(keys_file, 'r') as f:
            for name, value in csv.reader(f, delimiter='='):
                self.api_keys[name] = value

    def get_toxicity(self, message):
        """ API documentation: https://support.perspectiveapi.com/s/about-the-api-methods """
        # dictinary: {'a':123,'b':234}
        request = {
            'comment': {
                'text': self._strip_entities_from_text(message),
            },
            'requestedAttributes': {attribute: {} for attribute in self.attributes},
            'spanAnnotations': True
        }

        try:
            analyze_result = self.service.comments().analyze(body=request).execute()
            return analyze_result
        except googleapiclient.errors.HttpError as e:
            # weirdly the error_details aren't set on the error object until after `_get_reason` is called
            # e._get_reason()
            # error_type = e.error_details[0]['errorType']
            # self.error_types_count[error_type] += 1
            print("- Skipping due to http error in Perspective API")
            return None

    def get_attribute_score(self, analysis, attribute):
        """ Helper to get the score of a particular attribute from the perspective API response """
        return analysis['attributeScores'][attribute]['summaryScore']['value']

    def _print_formatted_analysis(self, analysis, padding=''):
        for attribute in self.attributes:
            print(f"{padding}- {attribute}: {self.get_attribute_score(analysis, attribute)}")

    def _strip_entities_from_text(self, text):
        """
        - Removes the '#' hashtag prefix but keeps the hashtag text
        - Replaces all username handles with 'user'
        - Removes urls
        """
        text = text.replace('#', '')
        text = re.sub(r'@\w+', 'user', text)
        text = re.sub(r'http\S+', '', text)
        return text

    def print_formatted_analysis(self, analysis, padding=''):
        for attribute in self.attributes:
            print(f"{padding}- {attribute}: {self.get_attribute_score(analysis, attribute)}")

if __name__ == '__main__':
    a = perspective_client()
    a.main()
