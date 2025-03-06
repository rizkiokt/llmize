from rich import print
from rich.console import Console
import os

def pretty_print(prompt=None, response=None):

    sol_color_start = "[bold cyan]"
    sol_color_end =   "[ /bold cyan]"

    if prompt:
        formatted_prompt = prompt.replace('<sol>', f'{sol_color_start}<sol>').replace('</sol>', f'</sol>{sol_color_end}')
        print(f"[bold yellow]Prompt:[/bold yellow] {formatted_prompt}")
    if response:
        formatted_response = response.replace('<sol>', f'{sol_color_start}<sol>').replace('</sol>', f'</sol>{sol_color_end}')
        print(f"[bold green]Response:[/bold green] {formatted_response}")