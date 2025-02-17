from pathlib import Path

from invoke.context import Context
from invoke.tasks import task


@task
def clean(ctx: Context):
    """Remove build artifacts."""
    ctx.run("rm -rf dist/")
    ctx.run("rm -rf build/")
    ctx.run("rm -rf *.egg-info")


@task
def lint(ctx: Context):
    """Run code quality checks."""
    ctx.run("ruff check .")
    ctx.run("ruff format .")
    ctx.run("mypy .")


@task
def test(ctx: Context):
    """Run tests."""
    ctx.run("pytest")


@task
def verify(ctx: Context):
    """Run all checks before deployment."""
    lint(ctx)
    test(ctx)


@task
def install(ctx: Context):
    """Install dependencies using uv."""
    ctx.run("uv pip install -e .[dev]")


@task
def heroku_login(ctx: Context):
    """Ensure logged into Heroku CLI."""
    ctx.run("heroku auth:whoami || heroku login")


@task
def configure_env(ctx: Context):
    """Configure Heroku environment variables from .env file"""
    ctx.run("set -o allexport; source .env; set +o allexport")


@task
def migrate(ctx: Context):
    """Run database migrations."""
    ctx.run("python manage.py migrate")


@task
def makemigrations(ctx: Context):
    """Create new migrations based on model changes."""
    ctx.run("python manage.py makemigrations")


@task
def serve(ctx: Context):
    """Run the development server."""
    ctx.run("heroku local")


@task
def logs(ctx: Context):
    """Show Heroku logs."""
    ctx.run("heroku logs --tail")


@task
def scale(ctx: Context, dynos: int = 1):
    """Scale Heroku dynos."""
    ctx.run(f"heroku ps:scale web={dynos}")


@task
def collectstatic(ctx: Context):
    """Collect static files for production deployment."""
    static_root = Path("staticfiles")
    static_dir = Path("src/pyjams/static")

    # Ensure directories exist
    static_root.mkdir(exist_ok=True)

    # Clear existing files
    ctx.run(f"rm -rf {static_root}/*")

    # Copy static files
    if static_dir.exists():
        ctx.run(f"cp -r {static_dir}/* {static_root}/")
        print(f"Collected static files to {static_root}")
    else:
        print(f"Warning: Static directory {static_dir} not found")


@task(pre=[verify, migrate, collectstatic])
def deploy(ctx: Context) -> None:
    """Deploy to Heroku with static files and migrations."""
    ctx.run("git push heroku master")
