from pathlib import Path

import mkdocs_gen_files


PACKAGE_DIRS = (Path("unienv_interface"), Path("unienv_data"))
DOCS_DIR = Path("docs")
INDEX_PATH = Path("api", "index.md")
SUMMARY_PATH = Path("SUMMARY.md")
PREFERRED_DOC_ORDER = [
    Path("index.md"),
    Path("getting-started.md"),
    Path("concepts", "index.md"),
    Path("concepts", "environments.md"),
    Path("concepts", "spaces-and-backends.md"),
    Path("concepts", "world-composition.md"),
    Path("guides", "index.md"),
    Path("guides", "wrappers-and-transformations.md"),
    Path("guides", "replay-buffers-and-storage.md"),
    Path("guides", "dataset-integrations.md"),
    Path("guides", "development.md"),
]


def should_skip(path: Path) -> bool:
    name = path.name
    if name == "__main__.py":
        return True
    if name.startswith("_") and name != "__init__.py":
        return True
    if "third_party" in path.parts:
        return True
    return False


def is_documentable(path: Path) -> bool:
    return (Path(path.parts[0]) / "__init__.py").exists()


def humanize(name: str) -> str:
    return name.replace("_", " ").replace("-", " ").title()


def doc_nav_parts(path: Path) -> tuple[str, ...]:
    if path.name == "index.md" and len(path.parts) == 1:
        return ("Home",)
    if path.name == "index.md":
        return tuple(humanize(part) for part in path.parts[:-1])
    return tuple(humanize(part) for part in path.parts[:-1]) + (humanize(path.stem),)


site_nav = mkdocs_gen_files.Nav()
api_nav_by_package = {package_dir.name: mkdocs_gen_files.Nav() for package_dir in PACKAGE_DIRS}

preferred_doc_positions = {
    path: index for index, path in enumerate(PREFERRED_DOC_ORDER)
}

doc_paths = []
for doc_path in DOCS_DIR.rglob("*.md"):
    rel_path = doc_path.relative_to(DOCS_DIR)
    if rel_path == SUMMARY_PATH or rel_path.parts[0] == "api":
        continue
    doc_paths.append(rel_path)

doc_paths.sort(
    key=lambda rel_path: (
        preferred_doc_positions.get(rel_path, len(PREFERRED_DOC_ORDER)),
        rel_path.as_posix(),
    )
)

for rel_path in doc_paths:
    site_nav[doc_nav_parts(rel_path)] = rel_path.as_posix()

for package_dir in PACKAGE_DIRS:
    for source_path in sorted(package_dir.rglob("*.py")):
        if should_skip(source_path) or not is_documentable(source_path):
            continue

        module_path = source_path.with_suffix("")
        module_parts = list(module_path.parts)

        if module_parts[-1] == "__init__":
            module_parts = module_parts[:-1]
            doc_path = Path("api", *module_parts, "index.md")
        else:
            doc_path = Path("api", *module_parts).with_suffix(".md")

        identifier = ".".join(module_parts)
        api_nav_by_package[package_dir.name][tuple(module_parts)] = doc_path.relative_to("api").as_posix()

        with mkdocs_gen_files.open(doc_path, "w") as fd:
            fd.write(f"# `{identifier}`\n\n")
            fd.write(f"::: {identifier}\n")

        mkdocs_gen_files.set_edit_path(doc_path, source_path)


with mkdocs_gen_files.open(INDEX_PATH, "w") as index_file:
    index_file.write("# API Reference\n\n")
    index_file.write(
        "These pages are generated from importable Python modules and rendered "
        "from their docstrings with `mkdocstrings`.\n\n"
    )
    index_file.write(
        "Use the narrative guides for concepts and workflows, then drop into "
        "this section when you need class- or module-level details.\n\n"
    )
    for package_dir in PACKAGE_DIRS:
        package_name = package_dir.name
        index_file.write(f"## `{package_name}`\n\n")
        index_file.writelines(api_nav_by_package[package_name].build_literate_nav())
        index_file.write("\n")


with mkdocs_gen_files.open(SUMMARY_PATH, "w") as summary_file:
    summary_file.writelines(site_nav.build_literate_nav())
    summary_file.write("* [API Reference](api/index.md)\n")
    for package_dir in PACKAGE_DIRS:
        for line in api_nav_by_package[package_dir.name].build_literate_nav():
            summary_file.write("  " + line.replace("](", "](api/", 1))
