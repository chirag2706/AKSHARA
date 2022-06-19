import { connect } from "react-redux";
import Managemodel from "../components/Managemodel";
import Annotationcanvas from "../components/Annotationcanvas";
import Annotationlist from "../components/Annotationlist";
import { useHistory } from "react-router-dom";
import { useLocation } from "react-router-dom";
import { useParams } from "react-router-dom";
import { useEffect } from "react";
import { useState } from "react";
import { getAnnotations, getDocument } from "../scripts/document";

const Document = (props) => {
    let history = useHistory();
    const { projectId, documentId } = useParams();
    const [document, setDocument] = useState({});
    const [annotations, setAnnotations] = useState([]);
    const [loading, setLoading] = useState(true);
    const [validDocument, setValidDocument] = useState(true);

    if (!props.auth.token) {
        history.push("/login");
    }

    useEffect(() => {
        async function load() {
            setLoading(true);
            let response = await getDocument(props.auth.token, {
                project: projectId,
                document: documentId,
            });
            if (response) {
                setDocument(response.data);
                setValidDocument(true);
            } else {
                setValidDocument(false);
            }
            setLoading(false);
            let a_response = await getAnnotations(props.auth.token, {
                project: projectId,
                document: documentId,
            });
            if (a_response) {
                setAnnotations(
                    a_response.map((value) => {
                        return {
                            _id: value._id,
                            name: value.name,
                            x: value.topX,
                            y: value.topY,
                            width: value.bottomX - value.topX,
                            height: value.bottomY - value.topY,
                            isAntiPattern: value.is_antipattern,
                            groundTruth: value.ground_truth,
                            visible: true,
                            type: "rectangle"
                        };
                    })
                );
            }
        }
        load();
    }, []);

    const toggleAnnotationsVisibility = (e) => {
        setAnnotations(annotations.map((annotation) => ({...annotation, visible: e.target.checked})))
    }

    return (
        <div>
            {loading ? (
                <div style={{ paddingTop: "100px" }}>
                    <div className="d-flex justify-content-center">
                        <div className="spinner-border" role="status" style={{ color: "#4285F4" }}></div>
                    </div>
                    <div style={{ textAlign: "center", paddingTop: "20px" }}>Loading Document Details. Please wait</div>
                </div>
            ) : !validDocument ? (
                <div style={{ paddingTop: "100px" }}>
                    <div className="d-flex justify-content-center">
                        <i className="fa-solid fa-bomb fa-xl"></i>
                    </div>
                    <div style={{ textAlign: "center", paddingTop: "20px" }}>Sorry. Document cannot be found.</div>
                </div>
            ) : (
                <div>
                    <div className="row justify-content-around">
                        <div className="col-md-8">
                            <Annotationcanvas data={{ annotations, setAnnotations, documentImage: document.image, token: props.auth.token, projectId: projectId, documentId: documentId }} />
                        </div>
                        <div className="col-md-3">
                            <div className="card text-center mt-3">
                                <div className="card-header">Document Details</div>
                                <div className="card-body">
                                    <h5 className="card-title">{document.name}</h5>
                                    <p className="card-text">{document.description}</p>
                                    <button className="btn btn-primary">Edit</button>
                                </div>
                            </div>
                            <div className="card mt-3">
                                <div className="card-header text-center">Annotations</div>
                                <div className="card-body">
                                    <Annotationlist data={{ annotations, setAnnotations, projectId, documentId, token: props.auth.token }} />
                                </div>
                                <div className="card-footer">
                                <input type="checkbox" onChange={(e) => {toggleAnnotationsVisibility(e)}} defaultChecked="true"></input>
                                <span style={{ paddingLeft: "10px" }}>Toggle Annotations</span>
                                </div>
                            </div>
                        </div>
                    </div>
                    <Managemodel projectId={projectId} documentId={documentId}/>
                </div>
            )}
        </div>
    );
};

export default connect((state) => ({ auth: state.auth }))(Document);
