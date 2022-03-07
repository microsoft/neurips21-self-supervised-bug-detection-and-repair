from typing import Dict, Set, Union

import libcst as cst
from libcst.metadata import QualifiedName, QualifiedNameProvider


class TypeAnnotationQualifier(cst.CSTVisitor):
    """
    Convert annotations into their fully-qualified counterparts.

    For example, given the annotation `Dict[str, MypyFile]`,
    where `MypyFile` has been appropriately imported, the annotation is
    rewritten to the exact":
        `typing.Dict[builtins.str, mypy.nodes.MypyFile]`

    """

    METADATA_DEPENDENCIES = (QualifiedNameProvider,)

    def __init__(self):
        super().__init__()
        self.__stack = []
        self._in_cx = False
        self.fq_type_annotations = {}

    def visit_Name(self, node: cst.Name) -> bool:
        if not self._in_cx:
            return True
        qual_names: Set[QualifiedName] = self.get_metadata(QualifiedNameProvider, node)
        assert len(qual_names) > 0
        self.__stack.append(qual_names.pop().name)
        return False

    def visit_Attribute(self, node: cst.Attribute) -> bool:
        if not self._in_cx:
            return True
        qual_names: Set[QualifiedName] = self.get_metadata(QualifiedNameProvider, node)
        assert len(qual_names) > 0
        self.__stack.append(qual_names.pop().name)
        return False

    def visit_List(self, node: cst.List) -> bool:
        if not self._in_cx:
            return True

        elements = []
        for e in node.elements:
            e.visit(self)
            elements.append(self.__stack.pop())

        self.__stack.append(f"[{', '.join(elements)}]")
        return False

    def visit_SimpleString(self, node: cst.SimpleString) -> bool:
        if not self._in_cx:
            return True
        self.__stack.append(node.evaluated_value)
        return False

    def visit_Ellipsis(self, node: cst.Ellipsis) -> None:
        if self._in_cx:
            self.__stack.append("...")

    def visit_Subscript(self, node: cst.Subscript) -> bool:
        if not self._in_cx:
            return True
        node.value.visit(self)
        value_name = self.__stack.pop()
        elements = []
        for e in node.slice:
            e.visit(self)
            elements.append(self.__stack.pop())
        self.__stack.append(f"{value_name}[{', '.join(elements)}]")
        return False

    def visit_Annotation(self, node: cst.Annotation) -> bool:
        self._in_cx = True
        return True

    def leave_Annotation(self, original_node: cst.Annotation) -> None:
        self.fq_type_annotations[original_node] = self.__stack.pop()
        self._in_cx = False


class RemoveTypeAnnotations(cst.CSTTransformer):
    METADATA_DEPENDENCIES = (QualifiedNameProvider,)

    def __init__(
        self,
    ):
        super().__init__()
        self.annotations: Dict[str, cst.Annotation] = {}

    def leave_AnnAssign(
        self, original_node: cst.AnnAssign, updated_node: cst.AnnAssign
    ) -> Union[cst.Assign, cst.RemovalSentinel]:
        if updated_node.value is None:
            # This is an annotation (nothing being assigned)
            return cst.RemovalSentinel.REMOVE

        qual_names: Set[QualifiedName] = self.get_metadata(QualifiedNameProvider, original_node.target)
        for qname in qual_names:
            self.annotations[qname.name] = original_node.annotation
        return cst.Assign(targets=[cst.AssignTarget(target=updated_node.target)], value=updated_node.value)

    def leave_Param(self, original_node: cst.Param, updated_node: cst.Param) -> cst.Param:
        if original_node.annotation is not None:
            qual_names: Set[QualifiedName] = self.get_metadata(QualifiedNameProvider, original_node.name)
            for qname in qual_names:
                self.annotations[qname.name] = original_node.annotation

        return updated_node.with_changes(annotation=None)

    def leave_FunctionDef(self, original_node: cst.FunctionDef, updated_node: cst.FunctionDef) -> cst.FunctionDef:
        if original_node.returns is not None:
            qual_names: Set[QualifiedName] = self.get_metadata(QualifiedNameProvider, original_node)
            for qname in qual_names:
                self.annotations[qname.name] = original_node.returns
            return updated_node.with_changes(returns=None)
        return updated_node


if __name__ == "__main__":
    import sys

    with open(sys.argv[1]) as f:
        module: cst.Module = cst.parse_module(f.read())
    wrapper = cst.MetadataWrapper(module)
    type_annotation_remover = RemoveTypeAnnotations()
    result = wrapper.visit(type_annotation_remover)

    for qname, annot in type_annotation_remover.annotations.items():
        print(qname, module.code_for_node(annot.annotation))
