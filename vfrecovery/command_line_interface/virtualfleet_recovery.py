import click

from vfrecovery.command_line_interface.group_describe import cli_group_describe
from vfrecovery.command_line_interface.group_predict import cli_group_predict

@click.command(
    cls=click.CommandCollection,
    sources=[
        cli_group_describe,
        cli_group_predict,
    ],
    context_settings=dict(help_option_names=["-h", "--help"]),
)
@click.version_option(None, "-V", "--version", package_name="vfrecovery")
def base_command_line_interface():
    pass


def command_line_interface():
    base_command_line_interface(windows_expand_args=False)


if __name__ == "__main__":
    command_line_interface()
