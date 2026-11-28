"""Forward a panel to a reduction through the common DSL's fragment staging."""

from collections import Counter

from tvm import ir, tirx

from tilelang.hexagon.intrinsics.panel import PanelLayout


def forward_panel_reductions(mod):
    """Remove a private copy immediately consumed by a reduction in the same sequence."""

    def transform(func, _mod, _ctx):
        root = func.body
        if not isinstance(root, tirx.SBlockRealize):
            return func
        layouts = root.block.annotations.get("layout_map", {})
        panels = {buf.data for buf, layout in layouts.items() if isinstance(layout, PanelLayout) and buf.dtype in ("int8", "uint8")}
        if not panels or not any(buf.scope() == "local.fragment" and buf.dtype in ("int8", "uint8") for buf in layouts):
            return func

        @tirx.functor.visitor
        class Uses(tirx.PyStmtExprVisitor):
            def __init__(self):
                super().__init__()
                self.count = Counter()
                self.private = set()

            def visit_sblock_(self, block):
                self.private.update(b.data for b in block.alloc_buffers if b.scope() == "local.fragment")
                for view in block.match_buffers:
                    self.count[view.source.buffer.data] += 1
                super().visit_sblock_(block)

            def visit_buffer_load_(self, load):
                self.count[load.buffer.data] += 1
                super().visit_buffer_load_(load)

            def visit_buffer_store_(self, store):
                self.count[store.buffer.data] += 1
                super().visit_buffer_store_(store)

            def visit_var_(self, var):
                # A pointer use can observe the staging copy independently.
                self.count[var] += 1

        uses = Uses()
        uses.visit_stmt(func.body)

        def call(stmt, name):
            if isinstance(stmt, tirx.Evaluate) and isinstance(stmt.value, tirx.Call) and getattr(stmt.value.op, "name", "") == name:
                return stmt.value
            return None

        def rewrite(seq):
            result = []
            for stmt in seq.seq:
                reduce = call(stmt, "tl.tileop.reduce")
                copy = call(result[-1], "tl.tileop.copy") if result and reduce is not None else None
                if copy is not None:
                    src, tmp = copy.args[:2]
                    read = reduce.args[0]
                    buf = tmp.args[0].buffer
                    if (
                        src.args[0].buffer.data in panels
                        and buf.data in uses.private
                        and uses.count[buf.data] == 2
                        and not src.args[0].buffer.data.same_as(reduce.args[1].args[0].buffer.data)
                        and ir.structural_equal(src.args[2:], tmp.args[2:])
                        and ir.structural_equal(tmp.args[0], read.args[0])
                        and ir.structural_equal(tmp.args[2:], read.args[2:])
                        and src.args[0].buffer.dtype == buf.dtype
                        and not copy.annotations
                    ):
                        result.pop()
                        stmt = tirx.Evaluate(tirx.Call(reduce.dtype, reduce.op, [src, *reduce.args[1:]], reduce.annotations, reduce.span))
                result.append(stmt)
            return tirx.SeqStmt(result) if len(result) != len(seq.seq) else seq

        body = tirx.stmt_functor.ir_transform(func.body, None, rewrite, ["tirx.SeqStmt"])
        return func.with_body(body)

    return tirx.transform.prim_func_pass(transform, opt_level=0, name="tl.hexagon.ForwardPanelReductions")(mod)
