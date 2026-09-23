"""G5 launcher constants: the site literals are located by position in the launcher's argv, and
nothing derived from them alone (a salted or plain digest of a low-entropy value) is committed."""
from tools.equivalence import identities

SYNTHETIC_LAUNCHER = '''
def fleet():
    return subprocess.run(['gcloud','compute','tpus','tpu-vm','ssh','example-vm-1',
        '--zone=example-zone1-a','--worker=all','--dry-run','--command=true'],capture_output=True)
def worker(argv):
    return [*argv,'--coordinator-address','203.0.113.7:8476','--wall-seconds','60']
'''


def test_site_literals_are_found_by_position():
    assert identities.launcher_site_literals(SYNTHETIC_LAUNCHER) == dict(
        coordinator="203.0.113.7:8476", zone="example-zone1-a", tpu_name="example-vm-1")


def test_absent_site_literals_are_reported_as_absent():
    assert identities.launcher_site_literals("x = ['--wall-seconds', '60']\n") == dict(
        coordinator=None, zone=None, tpu_name=None)


def test_no_expected_launcher_digests_are_committed():
    assert not any(name.startswith(("LAUNCHER_LITERAL", "LITERAL_SALT")) for name in vars(identities))
