import io
import json
import unittest
import urllib.error
from unittest.mock import patch

from agent import completion, ProviderError, tool_rejection_detail


class ProviderTests(unittest.TestCase):
    def test_long_failed_generation_preserves_recovery_metadata_without_leaking(self):
        body=json.dumps({'error':{'code':'tool_use_failed','message':'Tool validation failed.','failed_generation':'private reasoning '*1000}}).encode()
        error=urllib.error.HTTPError('https://api.groq.com',400,'bad tool',{},io.BytesIO(body))
        with patch('agent.urllib.request.urlopen',side_effect=error):
            with self.assertRaises(ProviderError) as caught:completion('test-key','test-model',[])
        self.assertEqual(caught.exception.code,'tool_use_failed')
        self.assertEqual(caught.exception.http_status,400)
        self.assertNotIn('private reasoning',str(caught.exception))

    def test_provider_preserves_tool_failure_code(self):
        body=json.dumps({'error':{'code':'tool_use_failed','message':"parameters for tool build_plan did not match schema: errors: [additionalProperties 'holiday_factor' not allowed]",'failed_generation':'private reasoning test-key'}}).encode()
        error=urllib.error.HTTPError('https://api.groq.com',400,'bad tool',{},io.BytesIO(body))
        with patch('agent.urllib.request.urlopen',side_effect=error):
            with self.assertRaises(ProviderError) as caught: completion('test-key','test-model',[])
        self.assertEqual(caught.exception.code,'tool_use_failed')
        self.assertIn('build_plan',caught.exception.recovery_detail)
        self.assertIn('holiday_factor',caught.exception.recovery_detail)
        self.assertNotIn('private reasoning',caught.exception.recovery_detail)
        self.assertNotIn('test-key',caught.exception.recovery_detail)

    def test_unknown_diagnostics_do_not_expose_sensitive_provider_data(self):
        message="parameters for tool unknown did not match schema: errors: [additionalProperties 'gsk_secret' not allowed] private reasoning"
        detail=tool_rejection_detail(message)
        self.assertNotIn('gsk_secret',detail)
        self.assertNotIn('private reasoning',detail)

    def test_rate_limit_retries_after_server_delay(self):
        error=urllib.error.HTTPError('https://api.groq.com',429,'limit',{'retry-after':'2'},io.BytesIO(b'{}'))
        response=io.BytesIO(b'{"choices":[{"message":{"content":"OK"}}]}')
        with patch('agent.urllib.request.urlopen',side_effect=[error,response]) as request, patch('agent.time.sleep') as sleep:
            self.assertEqual(completion('test-key','test-model',[])['content'],'OK')
        self.assertEqual(request.call_count,2)
        sleep.assert_called_once_with(2.25)

    def test_rate_limit_retry_is_bounded(self):
        errors=[urllib.error.HTTPError('https://api.groq.com',429,'limit',{'retry-after':'1'},io.BytesIO(b'{}')) for _ in range(3)]
        with patch('agent.urllib.request.urlopen',side_effect=errors) as request, patch('agent.time.sleep') as sleep:
            with self.assertRaises(ProviderError): completion('test-key','test-model',[])
        self.assertEqual(request.call_count,3)
        self.assertEqual(sleep.call_count,2)

    def test_application_signature_and_completion(self):
        response=io.BytesIO(json.dumps({'choices':[{'message':{'role':'assistant','content':'OK'}}]}).encode())
        with patch('agent.urllib.request.urlopen',return_value=response) as request:
            self.assertEqual(completion('test-key','test-model',[])['content'],'OK')
        headers=dict((k.lower(),v) for k,v in request.call_args.args[0].header_items())
        self.assertEqual(headers['user-agent'],'PromoPilot/1.0')
        self.assertEqual(headers['authorization'],'Bearer test-key')

    def test_errors_are_actionable_and_redact_key(self):
        cases=[
            (403,b'error code: 1010','client signature'),
            (403,json.dumps({'error':{'code':'model_blocked','message':'blocked test-key'}}).encode(),'model permissions'),
            (401,b'<html>unauthorized</html>','GROQ_API_KEY'),
            (429,b'{}','rate or quota'),
        ]
        for status,body,expected in cases:
            with self.subTest(status=status,body=body):
                error=urllib.error.HTTPError('https://api.groq.com',status,'error',{},io.BytesIO(body))
                with patch('agent.urllib.request.urlopen',side_effect=error):
                    with self.assertRaises(ProviderError) as caught:
                        completion('test-key','test-model',[])
                message=str(caught.exception)
                self.assertIn(expected,message)
                self.assertNotIn('test-key',message)
                self.assertNotIn('<html>',message)


if __name__=='__main__': unittest.main()
