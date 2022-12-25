# CounterBot

> ⚠ CounterBot may return offensive responses ⚠
> Note that this only demonstrates counterspeech generation for chatbots, and is not intended for commercial use.

[CounterBot](https://app-2swb3drj4a-uk.a.run.app/) is a chatbot designed to combat hate speech through generating counterspeech that challenges the hate narrative. It is built with Python 3.10, using [Flask](https://flask.palletsprojects.com/en/2.2.x/) as the web framework and [Gunicorn](https://docs.gunicorn.org/en/stable/) as the WSGI server. The application is built into a container with [Docker](https://www.docker.com/) and deployed on [Google Cloud Run](https://cloud.google.com/run/docs/overview/what-is-cloud-run).

**Counterspeech** is responding to hate with feedback that undermines hate speech, providing an enticing alternative to the content blocking and censorship system. The main advantage of counterspeech is that it preserves the freedom of speech. However, the effort of manually generating counterspeech prevents it from being effective at reducing hate at scale. By utilizing large language models, it becomes feasible to augment the process of detecting hate and generating counterspeech to extend to a larger use case e.g. reducing hate on social media platforms. 

<img width="585" alt="CounterBot Demo" src="https://user-images.githubusercontent.com/113187341/209463066-58e68fb6-a994-4510-a3ff-d5b9d352aee6.png">

A web demonstration of CounterBot can be found [here](https://app-2swb3drj4a-uk.a.run.app/).
