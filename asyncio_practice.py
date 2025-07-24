import time
import pdb as debugger
import asyncio
from abc import ABC, abstractmethod
from typing import Union, Awaitable, List
import aiohttp
import httpx
import json

pdb = debugger.set_trace
API_KEY = 'AIzaSyD6-qpR66dIIQVzJksxbgAha1Edty6S5r0'


class BaseYoutubeClient(ABC):
    base_url = 'https://www.googleapis.com/youtube/v3/search'

    @abstractmethod
    def fetch_video_link(self, text: str) -> Union[str, Awaitable[str]]:
        ...


class YoutubeClient(BaseYoutubeClient, httpx.Client):

    def __init__(self, api_key: str):
        self.api_key = api_key
        super().__init__()

    def fetch_video_link(self, text: str) -> str:
        search_params = {'part': 'snippet', 'q': f'{text}',
                         'type': 'video', 'key': self.api_key}
        r = self.get(self.base_url, params=search_params)
        search_results = json.loads(r.text)
        video_id = search_results['items'][0]['id']['videoId']
        video_link = f'https://www.youtube.com/watch?v={video_id}'
        return video_link


class AsyncYoutubeClient(BaseYoutubeClient):

    def __init__(self, api_key: str):
        self.session = aiohttp.ClientSession()
        self.api_key = api_key
        super().__init__()

    async def close_session(self):
        await self.session.close()

    # def __del__(self):
    #     asyncio.run(self.close_session())

    async def fetch_video_link(self, text: str) -> str:
        search_params = {'part': 'snippet', 'q': f'{text}',
                         'type': 'video', 'key': self.api_key}
        async with self.session.get(self.base_url, params=search_params) as r:
            fetched_data = await r.text()
            video_id = json.loads(fetched_data)['items'][0]['id']['videoId']
            video_link = f'https://www.youtube.com/watch?v={video_id}'
        return video_link


class HttpxAsyncYoutubeClient(BaseYoutubeClient):

    def __init__(self, api_key: str):
        """
        Since we want to reuse the client, we can't use a context manager that closes it.
        We need to use a loop to exert more control over when the client is closed.  
        """
        self.client = httpx.AsyncClient()
        self.loop = asyncio.get_event_loop()
        self.api_key = api_key
        super().__init__()

    async def close(self):
        # httpx.AsyncClient.aclose must be awaited!
        await self.client.aclose()

    def __del__(self):
        """
        A destructor is provided to ensure that the client and the event loop are closed at exit.
        """
        # Use the loop to call async close, then stop/close loop.
        self.loop.run_until_complete(self.close())
        self.loop.close()

    async def fetch_video_link(self, text: str) -> str:
        search_params = {'part': 'snippet', 'q': f'{text}',
                         'type': 'video', 'key': self.api_key}
        r = await self.client.get(self.base_url, params=search_params)
        search_results = json.loads(r.text)
        video_id = search_results['items'][0]['id']['videoId']
        video_link = f'https://www.youtube.com/watch?v={video_id}'
        return video_link

    def fetch_many_video_links(self, texts: List[str]) -> List[str]:
        coroutines = asyncio.gather(
            *(self.fetch_video_link(text) for text in texts))
        links = self.loop.run_until_complete(coroutines)
        return links


texts = ['TOEFL resources']


def time_execution(fn):
    def wrapper():
        t0 = time.time()
        fn()
        dt = time.time() - t0
        print(
            f"The function {fn.__name__} took {round(dt, 2)} seconds to execute")
    return wrapper


async def main():
    client = AsyncYoutubeClient(api_key=API_KEY)
    links = await asyncio.gather(*[client.fetch_video_link(text) for text in texts])
    print(links)
    await client.close_session()


@time_execution
def sync_fetch():
    client = YoutubeClient(API_KEY)
    links = [client.fetch_video_link(text) for text in texts]
    print(links)


@time_execution
def async_fetch():
    asyncio.run(main())


@time_execution
def httpx_async_fetch():
    client = HttpxAsyncYoutubeClient(api_key=API_KEY)
    links = client.fetch_many_video_links(texts)
    print(links)


# class A:
#     def __init__(self, a):
#         self.a = a

# class B:
#     def __init__(self):
#         pass

# class C(A, B):
#     def __init__(self, a):
#         A.__init__(self, a=a)
#         B.__init__(self)


httpx_async_fetch()
sync_fetch()
async_fetch()
