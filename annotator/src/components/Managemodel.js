import { connect } from "react-redux";
import "./../styles/profile.css";
import "./../styles/managemodel.css";
import { useState } from "react";
import { annotateDocument, createModelPool, getModelPools, saveModelPool, trainModel } from "../scripts/document";
import { useEffect } from "react";

const Managemodel = (props) => {
    let projectId = props.projectId;
    let documentId = props.documentId;

    const [modelpoolList, setModelpoolList] = useState([]);

    const [displayedModelPools, setDisplayedModelPools] = useState([]);
    const [selectedModelpools, setSelectedModelpools] = useState([]);

    const [modalModelPool, setModalModelPool] = useState({
        id: "",
        name: "",
        description: "",
        modelpool_list: [],
        pool_models: [],
    });

    const [trainingProgressMessage, setTPM] = useState("");
    const [trainingProgressPercentage, setTPP] = useState(0);
    const [trainModelMessage, setTMM] = useState("");

    const handleModelpoolsSearch = (e) => {
        setDisplayedModelPools(
            modelpoolList.filter((modelpool) => {
                if (modelpool.name.toLowerCase().includes(e.target.value) || modelpool.description.toLowerCase().includes(e.target.value) || modelpool.subdescription_list.toLowerCase().includes(e.target.value)) return modelpool;
            })
        );
    };

    const handleModelpoolSelect = (val, id) => {
        setModelpoolList(modelpoolList.map((modelpool) => (modelpool._id === id ? { ...modelpool, is_selected: val } : modelpool)));
        setDisplayedModelPools(displayedModelPools.map((modelpool) => (modelpool._id === id ? { ...modelpool, is_selected: val } : modelpool)));
        if (val) {
            var pools = selectedModelpools;
            pools.push(modelpoolList.find((modelpool) => modelpool._id === id));
            setSelectedModelpools(pools);
        } else {
            setSelectedModelpools(selectedModelpools.filter((modelpool) => modelpool._id !== id));
        }
    };

    const handleModalSubModelPool = (val, id) => {
        console.log(val)
        setModalModelPool({ ...modalModelPool, modelpool_list: modalModelPool.modelpool_list.map((modelpool) => (modelpool._id === id ? { ...modelpool, is_active: val } : modelpool)) });
    };

    const handleMoreInfo = (id) => {
        setModalModelPool(modelpoolList.find((modelpool) => modelpool._id === id));
        document.getElementById("modal-toggle").click();
    };

    function updateTrainingProgress(val) {
        if(val < 90) {
            setTimeout(() => {
                let perc = Math.floor(Math.random() * 11 + 10)
                if(trainingProgressPercentage === 100) {
                    val = 100;
                    setTPP(100);
                } else {
                    if(val + perc < 90) {
                        val += perc
                        setTPP(val);
                        if (val > 50) {
                            setTPM('Started Training...');
                        } else if(val > 20) {
                            setTPM('Collecting Annotations');
                        }
                    } else {
                        val = 90;
                        setTPP(90);
                        setTPM('Saving Model');
                    }
                    updateTrainingProgress(val);
                }
            }, 1000)
        }
    }

    const handleModelTrain = async () => {
        setTPM('Collecting data');
        setTPP(0);
        updateTrainingProgress(0);

        var model_name = document.getElementById("modelname-input").value;
        if (model_name.split(" ").join("") === "") {
            setTMM("field required");
        } else {
            setTMM("Started Training Process");
            model_name = model_name.split(" ").join(" ");
            let response = await trainModel(props.auth.token, {
                model_name: model_name,
                project: projectId
            });
            if(response) {
                setTPM('model successfully trained');
                setTMM('')
                setTPP(100);
            } else {
                setTMM("Training failed. Try again")
                setTPM('Something went wrong. Try again');
                setTPP(0);
            }
        }
    };

    const handleCreateModelPool = async () => {
        let title = document.getElementById('cm-title').value;
        let description = document.getElementById('cm-description').value;
        let status = await createModelPool(props.auth.token, {
            'name': title,
            'description': description,
            'project': projectId,
            'selected_modelpools': selectedModelpools
        })
        if (status) {
            console.log("Success")
            document.getElementById('cm-title').value = "";
            document.getElementById('cm-description').value = "";
            handleLoadModelPools();
        }
    }

    const handleModelPoolSave = async () => {
        let status_list = []
        for(let i=0;i<modalModelPool.modelpool_list.length;i++){
            status_list.push({
                main_modelpool: modalModelPool._id,
                sub_modelpool: modalModelPool.modelpool_list[i]._id,
                is_active: modalModelPool.modelpool_list[i].is_active
            })
        }
        let status = await saveModelPool(props.auth.token, status_list);
        if(status) {
            setModelpoolList(modelpoolList.map((modelpool) => {
                if(modelpool._id === modalModelPool._id) {
                    return modalModelPool
                } else {
                    return modelpool
                }
            })) 
        }
    }

    const handleLoadModelPools = async () => {
        let response = await getModelPools(props.auth.token, {
            'project': projectId
        });
        if(response) {
            setModelpoolList(response);
            setDisplayedModelPools(response);
            setSelectedModelpools([])
        }
    }

    const handleAnnotatedocument = async () => {
        let status = await annotateDocument(props.auth.token, {
            'project': projectId,
            'document': documentId,
            'selected_modelpools': selectedModelpools
        })
        if(status) {
            window.location.reload()
        }
    }

    return (
        <div className="container">
            <nav>
                <div className="nav nav-tabs document-navbar" id="nav-tab" role="tablist">
                    <button className="col-6 nav-link active" id="nav-train-tab" data-bs-toggle="tab" data-bs-target="#nav-train" type="button" role="tab" aria-controls="nav-train" aria-selected="true">
                        <div className="nav-button">Train Model</div>
                    </button>
                    <button className="col-6 nav-link" id="nav-auto-annotate-tab" data-bs-toggle="tab" data-bs-target="#nav-auto-annotate" type="button" role="tab" aria-controls="nav-auto-annotate" aria-selected="false" onClick={() => {handleLoadModelPools()}}>
                        <div className="nav-button">Auto Annotate</div>
                    </button>
                </div>
            </nav>
            <div className="tab-content" id="nav-tabContent">
                <div className="tab-pane fade show active" id="nav-train" role="tabpanel" aria-labelledby="nav-train-tab">
                    <div style={{ padding: "15px 10px" }}>
                        <div className="row justify-content-center progress-container">
                            <div className="col-8 col-md-7">
                                <div className="progress" style={{ height: "30px" }}>
                                    <div className="progress-bar progress-bar-striped progress-bar-animated " role="progressbar" aria-valuenow={trainingProgressPercentage} aria-valuemin="0" aria-valuemax="100" style={{ width: trainingProgressPercentage.toString() + "%"  }}>
                                        {trainingProgressPercentage}%
                                    </div>
                                </div>
                            </div>
                            <div id="progress-msg" className="col-8 col-md-7" style={{ fontSize: "14px", color: "dimgrey", paddingTop: "5px" }}>
                                {trainingProgressMessage}
                            </div>
                        </div>
                        <div className="row justify-content-center">
                            <div className="col-8 col-md-6">
                                <input id="modelname-input" placeholder="Enter model name" maxLength={24} />
                                <div className="modelinput-errormsg">{trainModelMessage}</div>
                            </div>
                            <div className="col-4 col-md-2" style={{ textAlign: "center" }}>
                                <button id="train-button" onClick={() => {handleModelTrain()}}>
                                    Train
                                </button>
                            </div>
                        </div>
                    </div>
                </div>
                <div className="tab-pane fade" id="nav-auto-annotate" role="tabpanel" aria-labelledby="nav-auto-annotate-tab">
                    <button id="modal-toggle" type="button" className="btn btn-primary" data-bs-toggle="modal" data-bs-target="#modelpoolModal" style={{ display: "none" }}></button>

                    <div className="modal fade" id="modelpoolModal" tabIndex="-1" aria-labelledby="modelpoolTitle" aria-hidden="true">
                        <div className="modal-dialog modal-lg">
                            <div className="modal-content">
                                <div className="modal-header">
                                    <h5 className="modal-title" id="modelpoolTitle">
                                        {modalModelPool.name}
                                    </h5>
                                    <button type="button" className="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
                                </div>
                                <div className="modal-body" style={{ maxHeight: "300px", overflow: "auto" }}>
                                    <div className="modal-sub-title">Description</div>
                                    <div className="border-bottom pb-3">{modalModelPool.description}</div>
                                    <div className="modal-sub-title">Sub ModelPools</div>
                                    <div className="modal-subcontainer border-bottoms">
                                        <div className="row pills-container">
                                            {modalModelPool.modelpool_list.map((modelpool, i) => (
                                                <div key={i} className="col-auto selected-pill rounded">
                                                    <div className="row">
                                                        <div className="col-auto pill-title">{modelpool.name}</div>
                                                        <div className="col-auto" style={{ paddingTop: "3px" }}>
                                                            <input
                                                                className="form-check-input mt-0"
                                                                type="checkbox"
                                                                value=""
                                                                checked={modelpool.is_active}
                                                                onChange={(e) => {
                                                                    handleModalSubModelPool(e.target.checked, modelpool._id);
                                                                }}
                                                            />
                                                        </div>
                                                    </div>
                                                </div>
                                            ))}
                                        </div>
                                    </div>
                                    <div className="modal-sub-title">Models</div>
                                    <div className="modal-subcontainer border-bottoms">
                                        <div className="row pills-container">
                                            {modalModelPool.pool_models.map((model, i) => (
                                                <div key={i} className="col-auto selected-pill rounded" style={{ backgroundColor: "#ffdeeb", borderColor: "#7e191b" }}>
                                                    <div className="col-auto modelpill-title">{model.name}</div>
                                                </div>
                                            ))}
                                        </div>
                                    </div>
                                </div>
                                <div className="modal-footer">
                                    <button type="button" className="btn btn-danger" data-bs-dismiss="modal">
                                        Delete
                                    </button>
                                    <button type="button" className="btn btn-primary" onClick={() => {handleModelPoolSave()}}>
                                        Save
                                    </button>
                                </div>
                            </div>
                        </div>
                    </div>
                    <div className="row justify-content-center">
                        <div className="col-md-6">
                            <div className="container">
                                <div className="row justify-content-start search-bar-container">
                                    <div className="col-2 col-sm-2 col-md-1 search-icon">
                                        <i className="fas fa-search fa-lg"></i>
                                    </div>
                                    <div className="col-10 col-sm-10 col-md-11">
                                        <div className="row">
                                            <input
                                                type="text"
                                                className="search-bar"
                                                placeholder="Search ModelPool"
                                                onChange={(e) => {
                                                    handleModelpoolsSearch(e);
                                                }}
                                            />
                                        </div>
                                    </div>
                                </div>
                                <div className="row border border-top-0 border-dark shadow-lg">
                                    <div className="search-box-container">
                                        {displayedModelPools.map((modelpool, i) => (
                                            <div key={i} className="search-box-item border rounded shadow-sm" style={{ cursor: "default" }}>
                                                <div className="row">
                                                    <div className="col-1" style={{ textAlign: "center", padding: "0px", paddingTop: "35px", paddingLeft: "5px", paddingRight: "5px", margin: "0px" }}>
                                                        <input
                                                            className="form-check-input mt-0"
                                                            type="checkbox"
                                                            value=""
                                                            checked={modelpool.is_selected}
                                                            onChange={(e) => {
                                                                handleModelpoolSelect(e.target.checked, modelpool._id);
                                                            }}
                                                        />
                                                    </div>
                                                    <div className="col-11" style={{ padding: "0px", paddingRight: "10px" }}>
                                                        <div className="name">{modelpool.name}</div>
                                                        <div className="description">{modelpool.description}</div>
                                                        <div className="username-container">
                                                            <span className="username-helpertext rounded">sub modelpools:</span>
                                                            <span className="username">
                                                                {modelpool.modelpool_list.map((submodelpool, j) => (
                                                                    <span key={j} className="rounded submodelpool-pill" style={{ backgroundColor: submodelpool.is_active ? "#6fc0ab" : "#e2b1cd" }}>
                                                                        {submodelpool.name}
                                                                    </span>
                                                                ))}
                                                            </span>
                                                        </div>
                                                        <div
                                                            className="moreinfo-tag"
                                                            onClick={() => {
                                                                handleMoreInfo(modelpool._id);
                                                            }}
                                                        >
                                                            more info
                                                        </div>
                                                    </div>
                                                </div>
                                            </div>
                                        ))}
                                    </div>
                                </div>
                            </div>
                        </div>
                        <div className="col-md-6">
                            <div className="annotate-container">
                                <div className="annotate-section border-bottom border-dark">
                                    <div className="section-title">Selected ModelPools</div>
                                    <div className="selected-modelpools border border-dark">
                                        <div className="row pills-container">
                                            {selectedModelpools.map((modelpool, i) => (
                                                <div key={i} className="col-auto selected-pill rounded">
                                                    <div className="row">
                                                        <div className="col-auto pill-title">{modelpool.name}</div>
                                                        <div
                                                            className="col-auto cancel-sign"
                                                            onClick={() => {
                                                                handleModelpoolSelect(false, modelpool._id);
                                                            }}
                                                        >
                                                            &#x2717;
                                                        </div>
                                                    </div>
                                                </div>
                                            ))}
                                        </div>
                                    </div>
                                    <div>
                                        <button className="btn btn-success annotate-button" onClick={() => {handleAnnotatedocument()}}>Annotate</button>
                                    </div>
                                </div>
                                <div className="create-modelpool">
                                    <div className="row" style={{ margin: "10px 10px" }}>
                                        <div className="col-md-4">
                                            <div className="input-group mb-3">
                                                <span className="input-group-text input-helpertext">Title</span>
                                                <input id="cm-title" type="text" className="form-control input-field" maxLength={24} />
                                            </div>
                                        </div>
                                        <div className="col-md-8">
                                            <div className="input-group mb-3">
                                                <span className="input-group-text input-helpertext">Description</span>
                                                <input id="cm-description" type="text" className="form-control input-field" />
                                            </div>
                                        </div>
                                        <div className="hint">above selected modelpools are used to create the modelpool</div>
                                    </div>
                                    <div>
                                        <button className="btn btn-success create-button" onClick={() => {handleCreateModelPool()}}>Create ModelPool</button>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default connect((state) => ({ auth: state.auth }))(Managemodel);
