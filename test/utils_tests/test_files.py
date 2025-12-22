from upscaler.utils.files import (dir_by_dir_walk, dir_by_dir_parallel_walk, get_closest_dir,
                                   prune_empty_folders_parallel, prune_empty_folders, rm_tree, get_size)


def test_dir_by_dir_walk(tmp_path):
    (tmp_path / "dir1").mkdir()
    (tmp_path / "dir2").mkdir()
    (tmp_path / "dir1" / "file1.txt").touch()
    (tmp_path / "dir2" / "file2.txt").touch()
    (tmp_path / "file3.txt").touch()

    files = list(dir_by_dir_walk(str(tmp_path)))

    assert len(files) == 3
    assert str(tmp_path / "file3.txt") in files[0]
    assert str(tmp_path / "dir1" / "file1.txt") in files[1]
    assert str(tmp_path / "dir2" / "file2.txt") in files[2]


def test_dir_by_dir_parallel_walk(tmp_path):
    (tmp_path / "dir1").mkdir()
    (tmp_path / "dir2").mkdir()
    (tmp_path / "dir1" / "file1.txt").touch()
    (tmp_path / "dir2" / "file2.txt").touch()
    (tmp_path / "file3.txt").touch()

    files = list(dir_by_dir_parallel_walk(str(tmp_path)))

    assert len(files) == 3
    assert str(tmp_path / "file3.txt") in files[0]
    assert str(tmp_path / "dir1" / "file1.txt") in files[1]
    assert str(tmp_path / "dir2" / "file2.txt") in files[2]


def test_get_closest_dir(tmp_path):
    file_path = tmp_path / "file.txt"
    file_path.touch()

    assert get_closest_dir(str(tmp_path)) == str(tmp_path)
    assert get_closest_dir(str(file_path)) == str(tmp_path)


def test_prune_empty_folders(tmp_path):
    (tmp_path / "empty").mkdir()
    (tmp_path / "not_empty").mkdir()
    (tmp_path / "not_empty" / "file.txt").touch()

    prune_empty_folders(str(tmp_path))

    assert not (tmp_path / "empty").exists()
    assert (tmp_path / "not_empty").exists()


def test_prune_empty_folders_parallel(tmp_path):
    (tmp_path / "empty").mkdir()
    (tmp_path / "not_empty").mkdir()
    (tmp_path / "not_empty" / "file.txt").touch()

    prune_empty_folders_parallel(str(tmp_path))

    assert not (tmp_path / "empty").exists()
    assert (tmp_path / "not_empty").exists()


def test_rm_tree(tmp_path):
    dir_to_remove = tmp_path / "to_remove"
    dir_to_remove.mkdir()
    (dir_to_remove / "file.txt").touch()

    rm_tree(str(dir_to_remove))

    assert not dir_to_remove.exists()


def test_get_size(tmp_path):
    file1 = tmp_path / "file1.txt"
    file2 = tmp_path / "dir" / "file2.txt"
    file2.parent.mkdir()
    file1.write_text("a" * 100)
    file2.write_text("b" * 50)

    assert get_size(str(tmp_path)) == 150
