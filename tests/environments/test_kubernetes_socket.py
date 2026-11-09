"""Test that KubernetesEnvironment uses a per-instance ApiClient.

The per-instance ApiClient eliminates cross-instance pool contention and
per-call TLS handshake overhead. execute() and copy_to() share the same
ApiClient (via self.v1_api), while cleanup() uses a fresh dedicated
ApiClient since the instance client's pool may hold half-closed
WebSocket connections.
"""

import hashlib
from unittest.mock import MagicMock, patch

import pytest

from mimoagent.environments import TransportError
from mimoagent.environments.kubernetes import KubernetesEnvironment


@pytest.fixture
def k8s_env():
    """KubernetesEnvironment with kube config stubbed."""
    with patch("mimoagent.environments.kubernetes.config.load_kube_config"):
        with patch("mimoagent.environments.kubernetes.client.ApiClient"):
            with patch("mimoagent.environments.kubernetes.client.CoreV1Api"):
                env = KubernetesEnvironment(image="test:latest")
                env.pod_name = "mimoagent-test1234"
                env.pod_started = True
                yield env


class TestExecutePerInstanceApiClient:
    """execute() must use self.v1_api (per-instance), not create per-call clients."""

    def test_execute_uses_instance_v1_api(self, k8s_env):
        with patch("mimoagent.environments.kubernetes.stream") as mock_stream:
            resp = MagicMock()
            resp.is_open.return_value = False
            resp.returncode = 0
            resp.peek_stdout.return_value = False
            resp.peek_stderr.return_value = False
            mock_stream.return_value = resp

            result = k8s_env.execute("echo hello")

        assert result["reason"] == "ok"
        # stream() was called with self.v1_api's exec method
        mock_stream.assert_called_once()
        assert mock_stream.call_args[0][0] == k8s_env.v1_api.connect_get_namespaced_pod_exec

    def test_execute_does_not_create_new_api_client(self, k8s_env):
        with patch("mimoagent.environments.kubernetes.client.ApiClient") as mock_api_client:
            with patch("mimoagent.environments.kubernetes.stream") as mock_stream:
                resp = MagicMock()
                resp.is_open.return_value = False
                resp.returncode = 0
                resp.peek_stdout.return_value = False
                resp.peek_stderr.return_value = False
                mock_stream.return_value = resp

                k8s_env.execute("echo a")
                k8s_env.execute("echo b")

        # No new ApiClient created during execute calls
        mock_api_client.assert_not_called()

    def test_execute_returns_transport_error_on_stream_failure(self, k8s_env):
        with patch(
            "mimoagent.environments.kubernetes.stream",
            side_effect=Exception("socket is already closed."),
        ):
            result = k8s_env.execute("echo hello")

        assert result == {
            "output": "Error opening exec stream: socket is already closed.",
            "returncode": None,
            "reason": "transport_error",
        }

    def test_execute_does_not_mutate_v1_api(self, k8s_env):
        original_v1_api = k8s_env.v1_api

        with patch(
            "mimoagent.environments.kubernetes.stream",
            side_effect=Exception("socket is already closed."),
        ):
            k8s_env.execute("echo hello")

        assert k8s_env.v1_api is original_v1_api


class TestCopyToPerInstanceApiClient:
    """copy_to() must use self.v1_api, with resp.close() in finally."""

    @staticmethod
    def _successful_response():
        resp = MagicMock()
        resp.is_open.return_value = False
        resp.returncode = 0
        resp.peek_stdout.return_value = False
        resp.peek_stderr.return_value = False
        return resp

    def test_copy_to_uses_instance_v1_api(self, k8s_env, tmp_path):
        src_file = tmp_path / "test.txt"
        src_file.write_text("hello")

        with patch("mimoagent.environments.kubernetes.stream") as mock_stream:
            resp = MagicMock()
            resp.is_open.return_value = False
            resp.returncode = 0
            resp.peek_stdout.return_value = False
            resp.peek_stderr.return_value = False
            mock_stream.return_value = resp

            k8s_env.copy_to(str(src_file), "/dest/test.txt")

        mock_stream.assert_called_once()
        assert mock_stream.call_args[0][0] == k8s_env.v1_api.connect_get_namespaced_pod_exec

    def test_copy_to_bounds_untar_read_with_head(self, k8s_env, tmp_path):
        """untar command must pipe through ``head -c <archive_len>`` so the
        pod-side reader hits a real EOF.

        ``write_stdin("")`` does not close the exec stdin channel, so BusyBox
        tar (Alpine images) would otherwise block reading stdin forever and the
        exec returns rc=None. ``head -c N`` closes the pipe at exactly the
        archive size, giving tar EOF on both GNU and BusyBox.
        """
        src_file = tmp_path / "test.txt"
        src_file.write_text("hello")

        written = []
        with patch("mimoagent.environments.kubernetes.stream") as mock_stream:
            resp = MagicMock()
            resp.is_open.return_value = False
            resp.returncode = 0
            resp.peek_stdout.return_value = False
            resp.peek_stderr.return_value = False
            resp.write_stdin = lambda d: written.append(d)
            mock_stream.return_value = resp

            k8s_env.copy_to(str(src_file), "/dest/test.txt")

        cmd = mock_stream.call_args.kwargs["command"]
        cmd_str = cmd[-1]
        archive_len = len(written[0])
        assert f"head -c {archive_len} | tar xmf -" in cmd_str, cmd_str
        # the byte count handed to head must match the tar archive actually sent
        assert archive_len > 0

    def test_copy_to_closes_resp_in_finally(self, k8s_env, tmp_path):
        """If write_stdin raises after stream() opens, resp must still be closed."""
        src_file = tmp_path / "test.txt"
        src_file.write_text("hello")

        resp = MagicMock()
        resp.write_stdin.side_effect = Exception("broken pipe")
        resp.is_open.return_value = False
        resp.returncode = 0

        with patch("mimoagent.environments.kubernetes.stream", return_value=resp):
            with patch("time.sleep"):
                with pytest.raises(Exception, match="broken pipe"):
                    k8s_env.copy_to(str(src_file), "/dest/test.txt", max_retries=2)

        # 2 attempts, resp.close() called in finally each time
        assert resp.close.call_count == 2

    def test_copy_to_packs_tar_once_across_retries(self, k8s_env, tmp_path):
        src_file = tmp_path / "test.txt"
        src_file.write_text("hello")

        written = []

        def stream_side_effect(*args, **kwargs):
            stream_side_effect.calls += 1
            if stream_side_effect.calls <= 2:
                raise Exception("socket is already closed.")
            resp = MagicMock()
            resp.is_open.return_value = False
            resp.returncode = 0
            resp.peek_stdout.return_value = False
            resp.peek_stderr.return_value = False
            resp.write_stdin = lambda d: written.append(d)
            return resp

        stream_side_effect.calls = 0

        with patch("mimoagent.environments.kubernetes.stream", side_effect=stream_side_effect):
            with patch("time.sleep"):
                k8s_env.copy_to(str(src_file), "/dest/test.txt")

        # tar bytes only written on the successful (3rd) attempt
        assert len(written) >= 1
        assert isinstance(written[0], bytes) and len(written[0]) > 0

    def test_copy_to_exhausts_retries_on_persistent_error(self, k8s_env, tmp_path):
        src_file = tmp_path / "test.txt"
        src_file.write_text("hello")

        with patch(
            "mimoagent.environments.kubernetes.stream",
            side_effect=Exception("socket is already closed."),
        ) as mock_stream:
            with patch("time.sleep"):
                with pytest.raises(Exception, match="socket is already closed"):
                    k8s_env.copy_to(str(src_file), "/dest/test.txt", max_retries=3)

        assert mock_stream.call_count == 3

    def test_copy_to_automatically_chunks_large_regular_file(self, k8s_env, tmp_path, monkeypatch):
        src_file = tmp_path / "large.patch"
        src_file.write_bytes(b"large payload")
        monkeypatch.setattr(KubernetesEnvironment, "_COPY_TO_CHUNK_THRESHOLD_BYTES", src_file.stat().st_size)

        chunk_info = {
            "bytes": src_file.stat().st_size,
            "chunks": 1,
            "seconds": 0.1,
            "sha256": "abc",
        }
        with patch.object(k8s_env, "copy_file_chunked", return_value=chunk_info) as chunked:
            with patch("mimoagent.environments.kubernetes.stream") as mock_stream:
                k8s_env.copy_to(
                    str(src_file),
                    "/tmp/recalc_model.patch",
                    timeout=123,
                    max_retries=7,
                )

        chunked.assert_called_once_with(
            str(src_file),
            "/tmp/recalc_model.patch",
            timeout=123,
            max_retries=7,
        )
        mock_stream.assert_not_called()

    def test_copy_to_keeps_small_file_on_tar_stream(self, k8s_env, tmp_path, monkeypatch):
        src_file = tmp_path / "small.patch"
        src_file.write_bytes(b"small")
        # The ceiling applies to the archive, which is what actually crosses the
        # exec stream, and tar pads even a 5-byte file out to several blocks.
        monkeypatch.setattr(KubernetesEnvironment, "_COPY_TO_CHUNK_THRESHOLD_BYTES", 1 << 20)

        with patch.object(k8s_env, "copy_file_chunked") as chunked:
            with patch(
                "mimoagent.environments.kubernetes.stream",
                return_value=self._successful_response(),
            ) as mock_stream:
                k8s_env.copy_to(str(src_file), "/tmp/recalc_model.patch")

        chunked.assert_not_called()
        mock_stream.assert_called_once()

    def test_oversized_directory_goes_to_the_archive_path_not_copy_file_chunked(self, k8s_env, tmp_path, monkeypatch):
        """An oversized directory is staged as an archive, never sent to copy_file_chunked.

        copy_file_chunked writes one regular file at a byte offset; handing it a
        directory would land the tar stream on disk under the directory's name.
        It stays off that path, but it no longer streams either -- a single
        oversized stdin frame is what used to fail every retry identically.
        """
        src_dir = tmp_path / "payload"
        src_dir.mkdir()
        (src_dir / "file.txt").write_text("content")
        monkeypatch.setattr(KubernetesEnvironment, "_COPY_TO_CHUNK_THRESHOLD_BYTES", 0)

        with patch.object(k8s_env, "copy_file_chunked") as chunked:
            with patch.object(k8s_env, "_stage_and_extract_archive", return_value={"chunks": 1}) as staged:
                k8s_env.copy_to(str(src_dir), "/tmp/payload")

        chunked.assert_not_called()
        staged.assert_called_once()
        assert staged.call_args.args[1] == "/tmp"

    def test_oversized_symlink_goes_to_the_archive_path_not_copy_file_chunked(self, k8s_env, tmp_path, monkeypatch):
        """A symlink is archived so it lands as a link, not as a copy of its target."""
        target = tmp_path / "target.patch"
        target.write_bytes(b"large")
        src_link = tmp_path / "large.patch"
        src_link.symlink_to(target)
        monkeypatch.setattr(KubernetesEnvironment, "_COPY_TO_CHUNK_THRESHOLD_BYTES", 0)

        with patch.object(k8s_env, "copy_file_chunked") as chunked:
            with patch.object(k8s_env, "_stage_and_extract_archive", return_value={"chunks": 1}) as staged:
                k8s_env.copy_to(str(src_link), "/tmp/large.patch")

        chunked.assert_not_called()
        staged.assert_called_once()

    def test_oversized_file_to_a_directory_destination_goes_to_the_archive_path(self, k8s_env, tmp_path, monkeypatch):
        """A destination with no basename has no file to write at an offset, so it archives."""
        src_file = tmp_path / "large.patch"
        src_file.write_bytes(b"large")
        monkeypatch.setattr(KubernetesEnvironment, "_COPY_TO_CHUNK_THRESHOLD_BYTES", 0)

        with patch.object(k8s_env, "copy_file_chunked") as chunked:
            with patch.object(k8s_env, "_stage_and_extract_archive", return_value={"chunks": 1}) as staged:
                k8s_env.copy_to(str(src_file), "/tmp/")

        chunked.assert_not_called()
        staged.assert_called_once()

    def test_copy_file_chunked_supports_relative_destination(self, k8s_env, tmp_path):
        src_file = tmp_path / "patch.diff"
        src_file.write_bytes(b"patch")
        src_file.chmod(0o751)
        digest = hashlib.sha256(b"patch").hexdigest()
        k8s_env.execute = MagicMock(
            side_effect=[
                {"returncode": 0, "output": ""},
                {"returncode": 0, "output": digest},
                {"returncode": 0, "output": ""},
            ]
        )

        with patch.object(k8s_env, "_write_chunk"):
            k8s_env.copy_file_chunked(str(src_file), "recalc_model.patch")

        first_command = k8s_env.execute.call_args_list[0].args[0]
        mode_command = k8s_env.execute.call_args_list[2].args[0]
        assert first_command.startswith("mkdir -p . && : > recalc_model.patch")
        assert mode_command == "chmod 751 recalc_model.patch"


class TestCopyToOversizedDirectory:
    """An oversized directory must be chunked, not streamed in one frame.

    `copy_to` only ever chunked large *regular files*; a directory was packed
    into one in-memory tar and pushed through a single `write_stdin`. Above
    roughly 33 MB that write dies mid-transfer, and the retry loop re-sends the
    identical oversized frame, so a large workspace burned its whole retry
    budget and the rollout was dropped as an infra failure. These pin the
    framing that replaces it.
    """

    @staticmethod
    def _env_with_execute(k8s_env, digest_source):
        """Stub `execute` so the staged-archive path can run without a cluster.

        `digest_source` is called with the command and returns the sha256 the pod
        should report, which is how a truncated staging is simulated.
        """
        seen: list[str] = []

        def fake_execute(command, *args, **kwargs):
            seen.append(command)
            if command.startswith("sha256sum"):
                return {"returncode": 0, "output": digest_source()}
            return {"returncode": 0, "output": ""}

        k8s_env.execute = MagicMock(side_effect=fake_execute)
        return seen

    def test_archive_is_sent_in_bounded_chunks_and_unpacked_from_the_staged_file(self, k8s_env, tmp_path, monkeypatch):
        src_dir = tmp_path / "workspace"
        src_dir.mkdir()
        (src_dir / "big.bin").write_bytes(b"\0" * (300 << 10))
        # Force the oversized route, and make the chunk small enough that one
        # archive spans several of them.
        monkeypatch.setattr(KubernetesEnvironment, "_COPY_TO_CHUNK_THRESHOLD_BYTES", 64 << 10)
        monkeypatch.setattr(KubernetesEnvironment, "_COPY_TO_CHUNK_BYTES", 64 << 10)
        monkeypatch.setattr(KubernetesEnvironment, "_COPY_TO_COMPRESS_MIN_BYTES", 1 << 30)  # keep it uncompressed

        uploaded: list[tuple[int, bytes]] = []
        commands = self._env_with_execute(
            k8s_env, lambda: hashlib.sha256(b"".join(c for _, c in sorted(uploaded))).hexdigest()
        )

        def fake_write_chunk(quoted_dest, chunk, idx, chunk_bytes, **kwargs):
            assert len(chunk) <= chunk_bytes, "a chunk must never exceed the configured size"
            uploaded.append((idx, chunk))

        with patch.object(k8s_env, "_write_chunk", side_effect=fake_write_chunk):
            with patch("mimoagent.environments.kubernetes.stream") as mock_stream:
                k8s_env.copy_to(str(src_dir), "/work/workspace")

        # Nothing went through an exec stdin frame: that is the failure mode.
        mock_stream.assert_not_called()
        assert len(uploaded) > 1, "the archive should have been split"
        assert [idx for idx, _ in uploaded] == list(range(len(uploaded))), "offsets must be dense and ordered"

        # The archive is unpacked from the staged file, and the staging is removed.
        untar = [c for c in commands if " -C /work" in c]
        assert len(untar) == 1 and untar[0].startswith("tar xmf /tmp/_copyto_")
        assert any(c.startswith("rm -f /tmp/_copyto_") for c in commands)

    def test_a_truncated_staging_is_rejected_before_anything_is_unpacked(self, k8s_env, tmp_path, monkeypatch):
        """sha256 is the gate: a short archive must not be extracted.

        Extracting a truncated tar yields a half-populated workspace, which the
        agent then fails against for reasons that look like the task.
        """
        src_dir = tmp_path / "workspace"
        src_dir.mkdir()
        (src_dir / "big.bin").write_bytes(b"\0" * (200 << 10))
        monkeypatch.setattr(KubernetesEnvironment, "_COPY_TO_CHUNK_THRESHOLD_BYTES", 64 << 10)
        monkeypatch.setattr(KubernetesEnvironment, "_COPY_TO_COMPRESS_MIN_BYTES", 1 << 30)

        commands = self._env_with_execute(k8s_env, lambda: hashlib.sha256(b"truncated").hexdigest())

        with patch.object(k8s_env, "_write_chunk"):
            with pytest.raises(TransportError, match="sha256 mismatch"):
                k8s_env.copy_to(str(src_dir), "/work/workspace")

        assert not any(c.startswith("tar ") for c in commands), "must not unpack an unverified archive"
        assert any(c.startswith("rm -f /tmp/_copyto_") for c in commands), "staging must be cleaned up"

    def test_a_large_archive_is_compressed_and_unpacked_with_z(self, k8s_env, tmp_path, monkeypatch):
        """Compression is the speed lever: fewer bytes, and often no chunking at all."""
        src_dir = tmp_path / "workspace"
        src_dir.mkdir()
        # Highly compressible, so the archive drops back under the ceiling and
        # the copy returns to the single-exec path.
        (src_dir / "big.bin").write_bytes(b"\0" * (4 << 20))
        monkeypatch.setattr(KubernetesEnvironment, "_COPY_TO_COMPRESS_MIN_BYTES", 1 << 20)
        monkeypatch.setattr(KubernetesEnvironment, "_COPY_TO_CHUNK_THRESHOLD_BYTES", 1 << 20)
        k8s_env._pod_tar_gzip_ok = True

        written: list[bytes] = []
        with patch("mimoagent.environments.kubernetes.stream") as mock_stream:
            resp = MagicMock()
            resp.is_open.return_value = False
            resp.returncode = 0
            resp.peek_stdout.return_value = False
            resp.peek_stderr.return_value = False
            resp.write_stdin = lambda d: written.append(d)
            mock_stream.return_value = resp

            k8s_env.copy_to(str(src_dir), "/work/workspace")

        payload = written[0]
        assert payload[:2] == b"\x1f\x8b", "the archive should be gzip framed"
        assert len(payload) < (1 << 20), "4 MB of zeros must compress under the ceiling"
        command = mock_stream.call_args.kwargs["command"][-1]
        assert f"head -c {len(payload)}" in command, "the pod-side read must be bounded by the real length"
        assert "tar xzmf -" in command, "a gzip archive needs the z flag to unpack"

    def test_a_pod_without_gzip_still_gets_a_plain_archive(self, k8s_env, tmp_path, monkeypatch):
        """The harness installers install gzip on demand, so some images lack it.

        Compression is an optimization; losing it must not break the copy.
        """
        src_dir = tmp_path / "workspace"
        src_dir.mkdir()
        (src_dir / "big.bin").write_bytes(b"\0" * (2 << 20))
        monkeypatch.setattr(KubernetesEnvironment, "_COPY_TO_COMPRESS_MIN_BYTES", 1 << 20)
        monkeypatch.setattr(KubernetesEnvironment, "_COPY_TO_CHUNK_THRESHOLD_BYTES", 1 << 30)
        k8s_env._pod_tar_gzip_ok = False

        written: list[bytes] = []
        with patch("mimoagent.environments.kubernetes.stream") as mock_stream:
            resp = MagicMock()
            resp.is_open.return_value = False
            resp.returncode = 0
            resp.peek_stdout.return_value = False
            resp.peek_stderr.return_value = False
            resp.write_stdin = lambda d: written.append(d)
            mock_stream.return_value = resp

            k8s_env.copy_to(str(src_dir), "/work/workspace")

        assert written[0][:2] != b"\x1f\x8b"
        command = mock_stream.call_args.kwargs["command"][-1]
        assert "tar xmf -" in command and "xzmf" not in command

    def test_the_gzip_probe_runs_once_per_pod(self, k8s_env):
        """It is one tiny exec, but it must not become one per copy."""
        k8s_env.execute = MagicMock(return_value={"returncode": 0, "output": "yes"})

        assert k8s_env._pod_supports_tar_gzip() is True
        assert k8s_env._pod_supports_tar_gzip() is True

        assert k8s_env.execute.call_count == 1


class TestCleanupDedicatedApiClient:
    """cleanup() uses a fresh ApiClient and closes the instance client first."""

    def test_cleanup_closes_instance_api_client(self, k8s_env):
        mock_instance_client = MagicMock()
        k8s_env._api_client = mock_instance_client

        with patch("mimoagent.environments.kubernetes.client.Configuration.get_default_copy"):
            with patch("mimoagent.environments.kubernetes.client.ApiClient"):
                mock_v1 = MagicMock()
                with patch("mimoagent.environments.kubernetes.client.CoreV1Api", return_value=mock_v1):
                    from kubernetes.client import ApiException

                    mock_v1.delete_namespaced_pod.side_effect = ApiException(status=404)
                    k8s_env.cleanup()

        # Instance client closed before delete cycle
        mock_instance_client.close.assert_called_once()
        assert k8s_env._api_client is None
