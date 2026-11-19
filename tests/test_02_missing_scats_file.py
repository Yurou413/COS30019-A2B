import app


def test_missing_scats_file_uses_fallback_sites():
    sites, used_fallback = app.load_scats_sites(
        "data/does-not-exist-for-test.xls"
    )

    assert used_fallback
    assert 2000 in sites["scats_id"].tolist()
    assert not sites.empty
