# Adapted for DispatchEvolve; see THIRD_PARTY_NOTICES.md for origin and changes.
"""
Main controller for DispatchEvolve Genetic Optimizer
"""

import asyncio
import json
import logging
import os
import pickle
import random
import shutil
import signal
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from dispatchevolve.optimizer.genetic.config import Config, load_config
from dispatchevolve.optimizer.genetic.database import Program, ProgramDatabase
from dispatchevolve.optimizer.genetic.evaluator import Evaluator
from dispatchevolve.optimizer.genetic.evolution_trace import EvolutionTracer
from dispatchevolve.optimizer.genetic.llm.ensemble import LLMEnsemble
from dispatchevolve.optimizer.genetic.process_parallel import ProcessParallelController
from dispatchevolve.optimizer.genetic.prompt.sampler import PromptSampler
from dispatchevolve.optimizer.genetic.utils.code_utils import extract_code_language
from dispatchevolve.optimizer.genetic.utils.format_utils import format_improvement_safe, format_metrics_safe

logger = logging.getLogger(__name__)

CHECKPOINT_COMPLETE_FILE = "checkpoint_complete.json"
CHECKPOINT_RNG_FILE = "rng_state.pkl"
_GENETIC_HANDLER_MARKER = "_dispatchevolve_genetic_handler"


def is_complete_checkpoint(path: Union[str, Path]) -> bool:
    """Return whether *path* is an atomically published optimizer checkpoint."""
    checkpoint = Path(path)
    marker = checkpoint / CHECKPOINT_COMPLETE_FILE
    metadata = checkpoint / "metadata.json"
    rng_path = checkpoint / CHECKPOINT_RNG_FILE
    if (
        not checkpoint.is_dir()
        or not marker.is_file()
        or not metadata.is_file()
        or not rng_path.is_file()
    ):
        return False
    try:
        payload = json.loads(marker.read_text(encoding="utf-8"))
        iteration = int(checkpoint.name.rsplit("_", 1)[-1])
        with rng_path.open("rb") as stream:
            rng_state = pickle.load(stream)
        if not isinstance(rng_state, dict) or set(rng_state) != {"python", "numpy"}:
            return False
        random.Random().setstate(rng_state["python"])

        import numpy as np

        np.random.RandomState().set_state(rng_state["numpy"])
    except Exception:
        # Any ordinary decoding or state-validation failure makes the
        # checkpoint non-resumable. Process-control BaseExceptions still
        # propagate normally.
        return False
    return payload.get("status") == "complete" and payload.get("iteration") == iteration


def prune_complete_checkpoints(
    checkpoint_root: Union[str, Path], *, keep: int = 1
) -> tuple[Path, ...]:
    """Remove superseded complete checkpoints and return those retained.

    Incomplete checkpoints are deliberately left untouched for diagnosis.  Call
    this only after the caller has durably materialized any history it needs
    from the iteration checkpoints.
    """
    if isinstance(keep, bool) or not isinstance(keep, int) or keep <= 0:
        raise ValueError("keep must be a positive integer")
    root = Path(checkpoint_root)
    complete = sorted(
        (path for path in root.glob("checkpoint_*") if is_complete_checkpoint(path)),
        key=lambda path: int(path.name.rsplit("_", 1)[-1]),
    )
    retained = tuple(complete[-keep:])
    for path in complete[:-keep]:
        shutil.rmtree(path)
    return retained


def _format_metrics(metrics: Dict[str, Any]) -> str:
    """Safely format metrics, handling both numeric and string values"""
    formatted_parts = []
    for name, value in metrics.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            try:
                formatted_parts.append(f"{name}={value:.4f}")
            except (ValueError, TypeError):
                formatted_parts.append(f"{name}={value}")
        else:
            formatted_parts.append(f"{name}={value}")
    return ", ".join(formatted_parts)


def _format_improvement(improvement: Dict[str, Any]) -> str:
    """Safely format improvement metrics"""
    formatted_parts = []
    for name, diff in improvement.items():
        if isinstance(diff, (int, float)) and not isinstance(diff, bool):
            try:
                formatted_parts.append(f"{name}={diff:+.4f}")
            except (ValueError, TypeError):
                formatted_parts.append(f"{name}={diff}")
        else:
            formatted_parts.append(f"{name}={diff}")
    return ", ".join(formatted_parts)


class GeneticOptimizer:
    """
    Main controller for DispatchEvolve Genetic Optimizer

    Orchestrates the evolution process, coordinating between the prompt sampler,
    LLM ensemble, evaluator, and program database.

    Features:
    - Tracks the absolute best program across evolution steps
    - Ensures the best solution is not lost during the MAP-Elites process
    - Always includes the best program in the selection process for inspiration
    - Maintains detailed logs and metadata about improvements
    """

    def __init__(
        self,
        initial_program_path: str,
        evaluation_file: str,
        config: Config,
        output_dir: Optional[str] = None,
    ):
        # Load configuration (loaded in main_async)
        self.config = config

        # Set up output directory
        self.output_dir = output_dir or os.path.join(
            os.path.dirname(initial_program_path), "genetic_optimizer_output"
        )
        os.makedirs(self.output_dir, exist_ok=True)

        self._setup_logging()

        self._setup_manual_mode_queue()

        # Set random seed for reproducibility if specified
        if self.config.random_seed is not None:
            import hashlib
            import random

            import numpy as np

            # Set global random seeds
            random.seed(self.config.random_seed)
            # NumPy legacy RNG accepts uint32 seeds; retain the full seed elsewhere.
            np.random.seed(self.config.random_seed % (2**32))

            # Create hash-based seeds for different components
            base_seed = str(self.config.random_seed).encode("utf-8")
            llm_seed = int(hashlib.md5(base_seed + b"llm").hexdigest()[:8], 16) % (2**31)

            # Propagate seed to LLM configurations
            self.config.llm.random_seed = llm_seed
            for model_cfg in self.config.llm.models:
                if not hasattr(model_cfg, "random_seed") or model_cfg.random_seed is None:
                    model_cfg.random_seed = llm_seed
            for model_cfg in self.config.llm.evaluator_models:
                if not hasattr(model_cfg, "random_seed") or model_cfg.random_seed is None:
                    model_cfg.random_seed = llm_seed

            logger.info(f"Set random seed to {self.config.random_seed} for reproducibility")
            logger.debug(f"Generated LLM seed: {llm_seed}")

        # Load initial program
        self.initial_program_path = initial_program_path
        self.initial_program_code = self._load_initial_program()
        if not self.config.language:
            self.config.language = extract_code_language(self.initial_program_code)

        # Extract file extension from initial program
        self.file_extension = os.path.splitext(initial_program_path)[1]
        if not self.file_extension:
            # Default to .py if no extension found
            self.file_extension = ".py"
        else:
            # Make sure it starts with a dot
            if not self.file_extension.startswith("."):
                self.file_extension = f".{self.file_extension}"

        # Set the file_suffix in config (can be overridden in YAML)
        if not hasattr(self.config, "file_suffix") or self.config.file_suffix == ".py":
            self.config.file_suffix = self.file_extension

        # Initialize components
        self.llm_ensemble = LLMEnsemble(self.config.llm.models)
        self.llm_evaluator_ensemble = LLMEnsemble(self.config.llm.evaluator_models)

        self.prompt_sampler = PromptSampler(self.config.prompt)
        self.evaluator_prompt_sampler = PromptSampler(self.config.prompt)
        self.evaluator_prompt_sampler.set_templates("evaluator_system_message")

        # Pass random seed to database if specified
        if self.config.random_seed is not None:
            self.config.database.random_seed = self.config.random_seed

        self.config.database.novelty_llm = self.llm_ensemble
        self.database = ProgramDatabase(self.config.database)

        self.evaluator = Evaluator(
            self.config.evaluator,
            evaluation_file,
            self.llm_evaluator_ensemble,
            self.evaluator_prompt_sampler,
            database=self.database,
            suffix=Path(self.initial_program_path).suffix,
        )
        self.evaluation_file = evaluation_file

        logger.info(f"Initialized GeneticOptimizer with {initial_program_path}")

        # Initialize evolution tracer
        if self.config.evolution_trace.enabled:
            trace_output_path = self.config.evolution_trace.output_path
            if not trace_output_path:
                # Default to output_dir/evolution_trace.{format}
                trace_output_path = os.path.join(
                    self.output_dir, f"evolution_trace.{self.config.evolution_trace.format}"
                )

            self.evolution_tracer = EvolutionTracer(
                output_path=trace_output_path,
                format=self.config.evolution_trace.format,
                include_code=self.config.evolution_trace.include_code,
                include_prompts=self.config.evolution_trace.include_prompts,
                enabled=True,
                buffer_size=self.config.evolution_trace.buffer_size,
                compress=self.config.evolution_trace.compress,
            )
            logger.info(f"Evolution tracing enabled: {trace_output_path}")
        else:
            self.evolution_tracer = None

        # Initialize improved parallel processing components
        self.parallel_controller = None

    def _setup_logging(self) -> None:
        """Set up logging"""
        log_dir = self.config.log_dir or os.path.join(self.output_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)

        # Set up root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, self.config.log_level))

        # A Full Dispatch round can construct several sequential optimizers.
        # Remove only handlers owned by an earlier optimizer so its file and
        # console destinations do not receive every later record again.
        for handler in tuple(root_logger.handlers):
            if getattr(handler, _GENETIC_HANDLER_MARKER, False):
                root_logger.removeHandler(handler)
                handler.close()

        # Add file handler
        log_file = os.path.join(log_dir, f"evolution_{time.strftime('%Y%m%d_%H%M%S')}.log")
        file_handler = logging.FileHandler(log_file)
        setattr(file_handler, _GENETIC_HANDLER_MARKER, True)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        root_logger.addHandler(file_handler)

        # Add console handler
        console_handler = logging.StreamHandler()
        setattr(console_handler, _GENETIC_HANDLER_MARKER, True)
        console_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        root_logger.addHandler(console_handler)

        logger.info(f"Logging to {log_file}")

    def _setup_manual_mode_queue(self) -> None:
        """
        Set up manual task queue directory if llm.manual_mode is enabled

        Queue directory is always:
          <genetic_optimizer_output>/manual_tasks_queue

        The directory is cleared on controller start so the UI shows only tasks
        from the current run (no stale tasks after restart)
        """
        if not bool(getattr(self.config.llm, "manual_mode", False)):
            return

        qdir = (Path(self.output_dir).expanduser().resolve() / "manual_tasks_queue")

        # Clear stale tasks from previous runs
        if qdir.exists():
            shutil.rmtree(qdir)
        qdir.mkdir(parents=True, exist_ok=True)

        # Inject runtime-only queue dir into configs
        self.config.llm._manual_queue_dir = str(qdir)
        for model_cfg in self.config.llm.models:
            model_cfg._manual_queue_dir = str(qdir)
        for model_cfg in self.config.llm.evaluator_models:
            model_cfg._manual_queue_dir = str(qdir)

        logger.info(f"Manual mode enabled. Queue dir: {qdir}")

    def _load_initial_program(self) -> str:
        """Load the initial program from file"""
        with open(self.initial_program_path, "r") as f:
            return f.read()

    async def run(
        self,
        iterations: Optional[int] = None,
        target_score: Optional[float] = None,
        checkpoint_path: Optional[str] = None,
    ) -> Optional[Program]:
        """
        Run the evolution process with improved parallel processing

        Args:
            iterations: Maximum number of iterations (uses config if None)
            target_score: Target score to reach (continues until reached if specified)
            checkpoint_path: Path to resume from checkpoint

        Returns:
            Best program found
        """
        max_iterations = iterations or self.config.max_iterations

        # Determine starting iteration
        start_iteration = 0
        if checkpoint_path and os.path.exists(checkpoint_path):
            self._load_checkpoint(checkpoint_path)
            start_iteration = self.database.last_iteration + 1
            logger.info(f"Resuming from checkpoint at iteration {start_iteration}")
        else:
            start_iteration = self.database.last_iteration

        # Only add initial program if starting fresh (not resuming from checkpoint)
        should_add_initial = (
            start_iteration == 0
            and len(self.database.programs) == 0
            and not any(
                p.code == self.initial_program_code for p in self.database.programs.values()
            )
        )

        if should_add_initial:
            logger.info("Adding initial program to database")
            initial_program_id = str(uuid.uuid4())

            # Evaluate the initial program
            initial_metrics = await self.evaluator.evaluate_program(
                self.initial_program_code, initial_program_id
            )

            initial_program = Program(
                id=initial_program_id,
                code=self.initial_program_code,
                changes_description=self.config.prompt.initial_changes_description,
                language=self.config.language,
                metrics=initial_metrics,
                iteration_found=start_iteration,
            )

            self.database.add(initial_program)
            # Persist the paid initial evaluation before any mutation call. A
            # workflow resume can now continue without replaying that provider
            # call or evaluator side effect.
            self._save_checkpoint(0)

            # Check if combined_score is present in the metrics
            if "combined_score" not in initial_metrics:
                # Calculate average of numeric metrics
                numeric_metrics = [
                    v
                    for v in initial_metrics.values()
                    if isinstance(v, (int, float)) and not isinstance(v, bool)
                ]
                if numeric_metrics:
                    avg_score = sum(numeric_metrics) / len(numeric_metrics)
                    logger.warning(
                        f"⚠️  No 'combined_score' metric found in evaluation results. "
                        f"Using average of all numeric metrics ({avg_score:.4f}) for evolution guidance. "
                        f"For better evolution results, please modify your evaluator to return a 'combined_score' "
                        f"metric that properly weights different aspects of program performance."
                    )
        else:
            logger.info(
                f"Skipping initial program addition (resuming from iteration {start_iteration} "
                f"with {len(self.database.programs)} existing programs)"
            )

        # Initialize improved parallel processing
        try:
            self.parallel_controller = ProcessParallelController(
                self.config,
                self.evaluation_file,
                self.database,
                self.evolution_tracer,
                file_suffix=self.config.file_suffix,
            )

            # Set up signal handlers for graceful shutdown
            def signal_handler(signum, frame):
                logger.info(f"Received signal {signum}, initiating graceful shutdown...")
                self.parallel_controller.request_shutdown()

                # Set up a secondary handler for immediate exit if user presses Ctrl+C again
                def force_exit_handler(signum, frame):
                    logger.info("Force exit requested - terminating immediately")
                    import sys

                    sys.exit(0)

                signal.signal(signal.SIGINT, force_exit_handler)

            # A workflow may own several independent genetic runs concurrently.
            # Python permits signal registration only on the main thread; the
            # outer workflow remains responsible for process-level shutdown.
            if threading.current_thread() is threading.main_thread():
                signal.signal(signal.SIGINT, signal_handler)
                signal.signal(signal.SIGTERM, signal_handler)

            self.parallel_controller.start()

            # When starting from iteration 0, we've already done the initial program evaluation
            # So we need to adjust the start_iteration for the actual evolution
            evolution_start = start_iteration
            evolution_iterations = max(0, max_iterations - max(start_iteration - 1, 0))

            # If we just added the initial program at iteration 0, start evolution from iteration 1
            if should_add_initial and start_iteration == 0:
                evolution_start = 1
                # User expects max_iterations evolutionary iterations AFTER the initial program
                # So we don't need to reduce evolution_iterations

            # Run evolution with improved parallel processing and checkpoint callback
            if evolution_iterations > 0:
                await self._run_evolution_with_checkpoints(
                    evolution_start, evolution_iterations, target_score
                )

        finally:
            # Clean up parallel processing resources
            if self.parallel_controller:
                self.parallel_controller.stop()
                self.parallel_controller = None

            # Close evolution tracer
            if self.evolution_tracer:
                self.evolution_tracer.close()
                logger.info("Evolution tracer closed")

        # Get the best program
        best_program = None
        if self.database.best_program_id:
            best_program = self.database.get(self.database.best_program_id)
            logger.info(f"Using tracked best program: {self.database.best_program_id}")

        if best_program is None:
            best_program = self.database.get_best_program()
            logger.info("Using calculated best program (tracked program not found)")

        if best_program:
            if (
                hasattr(self, "parallel_controller")
                and self.parallel_controller
                and self.parallel_controller.early_stopping_triggered
            ):
                logger.info(
                    f"🛑 Evolution complete via early stopping. Best program has metrics: "
                    f"{format_metrics_safe(best_program.metrics)}"
                )
            else:
                logger.info(
                    f"Evolution complete. Best program has metrics: "
                    f"{format_metrics_safe(best_program.metrics)}"
                )
            self._save_best_program(best_program)
            return best_program
        else:
            logger.warning("No valid programs found during evolution")
            return None

    def _log_iteration(
        self,
        iteration: int,
        parent: Program,
        child: Program,
        elapsed_time: float,
    ) -> None:
        """
        Log iteration progress

        Args:
            iteration: Iteration number
            parent: Parent program
            child: Child program
            elapsed_time: Elapsed time in seconds
        """
        # Calculate improvement using safe formatting
        improvement_str = format_improvement_safe(parent.metrics, child.metrics)

        logger.info(
            f"Iteration {iteration+1}: Child {child.id} from parent {parent.id} "
            f"in {elapsed_time:.2f}s. Metrics: "
            f"{format_metrics_safe(child.metrics)} "
            f"(Δ: {improvement_str})"
        )

    def _save_checkpoint(self, iteration: int) -> None:
        """
        Save a checkpoint

        Args:
            iteration: Current iteration number
        """
        checkpoint_dir = Path(self.output_dir) / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoint_dir / f"checkpoint_{iteration}"
        if is_complete_checkpoint(checkpoint_path):
            logger.debug("Checkpoint %s is already complete", checkpoint_path)
            return

        staging_path = checkpoint_dir / f".checkpoint_{iteration}.{uuid.uuid4().hex}.tmp"
        staging_path.mkdir()

        try:
            # Save every component under an unpublished staging directory.
            self.database.save(str(staging_path), iteration)

            import numpy as np

            with (staging_path / CHECKPOINT_RNG_FILE).open("wb") as stream:
                pickle.dump(
                    {"python": random.getstate(), "numpy": np.random.get_state()},
                    stream,
                    protocol=pickle.HIGHEST_PROTOCOL,
                )

            # Save the best program found so far.
            best_program = None
            if self.database.best_program_id:
                best_program = self.database.get(self.database.best_program_id)
            else:
                best_program = self.database.get_best_program()

            if best_program:
                # Save the best program at this checkpoint.
                best_program_path = staging_path / f"best_program{self.file_extension}"
                best_program_path.write_text(best_program.code, encoding="utf-8")

                # Save metrics.
                best_program_info_path = staging_path / "best_program_info.json"
                with best_program_info_path.open("w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "id": best_program.id,
                            "generation": best_program.generation,
                            "iteration": best_program.iteration_found,
                            "current_iteration": iteration,
                            "metrics": best_program.metrics,
                            "language": best_program.language,
                            "timestamp": best_program.timestamp,
                            "saved_at": time.time(),
                        },
                        f,
                        indent=2,
                    )

                logger.info(
                    f"Saved best program at checkpoint {iteration} with metrics: "
                    f"{format_metrics_safe(best_program.metrics)}"
                )

            (staging_path / CHECKPOINT_COMPLETE_FILE).write_text(
                json.dumps(
                    {"status": "complete", "iteration": iteration, "saved_at": time.time()},
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            if checkpoint_path.exists():
                shutil.rmtree(checkpoint_path)
            os.replace(staging_path, checkpoint_path)
        except BaseException:
            shutil.rmtree(staging_path, ignore_errors=True)
            raise

        logger.info(f"Saved checkpoint at iteration {iteration} to {checkpoint_path}")

    def _load_checkpoint(self, checkpoint_path: str) -> None:
        """Load state from a checkpoint directory"""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint directory {checkpoint_path} not found")
        if not is_complete_checkpoint(checkpoint_path):
            raise ValueError(f"Checkpoint directory {checkpoint_path} is incomplete")

        logger.info(f"Loading checkpoint from {checkpoint_path}")
        self.database.load(checkpoint_path)
        rng_path = Path(checkpoint_path) / CHECKPOINT_RNG_FILE
        import numpy as np

        with rng_path.open("rb") as stream:
            rng_state = pickle.load(stream)
        random.setstate(rng_state["python"])
        np.random.set_state(rng_state["numpy"])
        logger.info(f"Checkpoint loaded successfully (iteration {self.database.last_iteration})")

    async def _run_evolution_with_checkpoints(
        self, start_iteration: int, max_iterations: int, target_score: Optional[float]
    ) -> None:
        """Run evolution with checkpoint saving support"""
        logger.info(f"Using island-based evolution with {self.config.database.num_islands} islands")
        self.database.log_island_status()

        # Run the evolution process with checkpoint callback
        await self.parallel_controller.run_evolution(
            start_iteration, max_iterations, target_score, checkpoint_callback=self._save_checkpoint
        )

        # Check if shutdown or early stopping was triggered
        if self.parallel_controller.shutdown_event.is_set():
            logger.info("Evolution stopped due to shutdown request")
            return
        elif self.parallel_controller.early_stopping_triggered:
            logger.info("Evolution stopped due to early stopping - saving final checkpoint")
            # Continue to save final checkpoint for early stopping

        # Save final checkpoint if needed
        # Note: start_iteration here is the evolution start (1 for fresh start, not 0)
        # max_iterations is the number of evolution iterations to run
        final_iteration = start_iteration + max_iterations - 1
        if final_iteration > 0 and final_iteration % self.config.checkpoint_interval == 0:
            self._save_checkpoint(final_iteration)

    def _save_best_program(self, program: Optional[Program] = None) -> None:
        """
        Save the best program

        Args:
            program: Best program (if None, uses the tracked best program)
        """
        # If no program is provided, use the tracked best program from the database
        if program is None:
            if self.database.best_program_id:
                program = self.database.get(self.database.best_program_id)
            else:
                # Fallback to calculating best program if no tracked best program
                program = self.database.get_best_program()

        if not program:
            logger.warning("No best program found to save")
            return

        best_dir = os.path.join(self.output_dir, "best")
        os.makedirs(best_dir, exist_ok=True)

        # Use the extension from the initial program file
        filename = f"best_program{self.file_extension}"
        code_path = os.path.join(best_dir, filename)

        with open(code_path, "w") as f:
            f.write(program.code)

        # Save complete program info including metrics
        info_path = os.path.join(best_dir, "best_program_info.json")
        with open(info_path, "w") as f:
            import json

            json.dump(
                {
                    "id": program.id,
                    "generation": program.generation,
                    "iteration": program.iteration_found,
                    "timestamp": program.timestamp,
                    "parent_id": program.parent_id,
                    "metrics": program.metrics,
                    "language": program.language,
                    "saved_at": time.time(),
                },
                f,
                indent=2,
            )

        logger.info(f"Saved best program to {code_path} with program info to {info_path}")
