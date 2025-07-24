import httpx
import pdb as debugger
from pathlib import Path
import time
from typing import Any
from dataclasses import dataclass
import asyncio


pdb = debugger.set_trace

API_KEY = '64a61483-f390-3fee-b4c9-45e120e8b5d6:fx'


class TranslationError(Exception):
    pass


def _check_file_format(filepath: Path) -> None:
    valid_suffixes = ['.docx', '.pptx', '.xlsx', '.pdf',
                      '.htm', '.html', '.txt', '.xlf', '.xliff']
    if filepath.suffix not in valid_suffixes:
        raise TranslationError(f'Unsupported file format: {filepath.suffix}')


def _save_downloaded_file(content: bytes, original_path: Path, target_lang: str) -> None:
    parent_path = original_path.parent.as_posix()
    target_filepath = (f'{parent_path}/{original_path.stem}_translated_to_'
                       f'{target_lang}{original_path.suffix}')
    with open(target_filepath, 'wb') as file:
        file.write(content)


@dataclass
class ClientUsageLimits:
    translated_characters: int
    character_limit: int

    @property
    def characters_left(self) -> int:
        return self.character_limit - self.translated_characters


class DeepLClient(httpx.Client):
    """
    DeepL httpx Client that supports asynchronus API calls
    """
    base_url = 'https://api-free.deepl.com/v2'

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.async_client = httpx.AsyncClient()
        super().__init__()

    def check_usage_limits(self) -> ClientUsageLimits:
        response = self.get(f'{self.base_url}/usage',
                            params={'auth_key': self.api_key})
        response.raise_for_status()
        json_resp = response.json()
        return ClientUsageLimits(json_resp['character_count'],
                                 json_resp['character_limit'])

    def translate(self, text: list[str], target_lang: str, **kwargs: Any) -> list[str]:
        data = {'text': text, 'target_lang': target_lang,
                'auth_key': self.api_key} | kwargs
        response = self.post(f'{self.base_url}/translate', data=data)
        json_resp = response.json()
        translations = json_resp.get('translations')
        if not translations:
            raise TranslationError(json_resp['message'])
        return [e['text'] for e in translations]

    def translate_file(self, filepath: str | Path, target_lang: str, **kwargs: Any) -> None:
        if isinstance(filepath, str):
            filepath = Path(filepath)
        _check_file_format(filepath)

        file = open(filepath, mode='rb')
        data = {'target_lang': target_lang, 'auth_key': self.api_key} | kwargs

        # Uploading file for translation
        response = self.post(
            f'{self.base_url}/document', data=data, files={'upload_file': file})
        response.raise_for_status()
        json_resp = response.json()
        doc_id = json_resp['document_id']
        doc_data = {'document_id': doc_id,
                    'document_key': json_resp['document_key'], 'auth_key': self.api_key}
        status = 'translating'

        while status != 'done':
            time.sleep(5)
            status = self._check_document_status(
                doc_id=doc_id, doc_data=doc_data)
        # The document has been translated

        response = self.post(
            f'{self.base_url}/document/{doc_id}/result', data=doc_data)
        _save_downloaded_file(content=response.content,
                              original_path=filepath, target_lang=target_lang)

    def _check_document_status(self, doc_id: str, doc_data: dict[str, str]) -> str:
        response = self.post(
            f'{self.base_url}/document/{doc_id}', data=doc_data)
        json_resp = response.json()
        status = json_resp['status']
        if status == 'error':
            raise TranslationError(json_resp['error_message'])
        return status

    async def _async_check_document_status(self, async_client: httpx.AsyncClient,
                                           doc_id: str, doc_data: dict[str, str]) -> str:
        response = await async_client.post(
            f'{self.base_url}/document/{doc_id}', data=doc_data)
        json_resp = response.json()
        status = json_resp['status']
        if status == 'error':
            raise TranslationError(json_resp['error_message'])
        return status

    async def async_translate_file(self, filepath: str | Path, target_lang: str,
                                   **kwargs: Any) -> None:
        if isinstance(filepath, str):
            filepath = Path(filepath)
        _check_file_format(filepath)

        file = open(filepath, mode='rb')
        data = {'target_lang': target_lang, 'auth_key': self.api_key} | kwargs
        async_client = httpx.AsyncClient()

        # Uploading file for translation
        response = await async_client.post(
            f'{self.base_url}/document', data=data, files={'upload_file': file})
        response.raise_for_status()
        json_resp = response.json()
        doc_id = json_resp['document_id']
        doc_data = {'document_id': doc_id,
                    'document_key': json_resp['document_key'], 'auth_key': self.api_key}
        status = 'translating'

        while status != 'done':
            time.sleep(5)
            status = await self._async_check_document_status(async_client=async_client,
                                                             doc_id=doc_id, doc_data=doc_data)
        # The document has been translated

        response = async_client.post(
            f'{self.base_url}/document/{doc_id}/result', data=doc_data)
        _save_downloaded_file(content=response.content,
                              original_path=filepath, target_lang=target_lang)

    def translate_multiple_files(self, filepaths: list[str | Path],
                                 target_lang: str, is_async: bool = False, **kwargs: Any) -> None:
        if not is_async:
            for filepath in filepaths:
                self.translate_file(
                    filepath, target_lang=target_lang, kwargs=kwargs)
        else:
            loop = asyncio.get_event_loop()
            coroutines = asyncio.gather(
                *(self.async_translate_file(filepath, target_lang=target_lang,
                                            kwargs=kwargs) for filepath in filepaths))
            loop.run_until_complete(coroutines)
            loop.close()


if __name__ == '__main__':
    client = DeepLClient(API_KEY)
    # filepath = r"C:\Users\fatih\python\files\file_1.docx"
    # client.translate_file(
    #     filepath=filepath, source_lang='PT', target_lang='ES')
    usage_data = client.check_usage_limits()
    print(usage_data)
    pdb()
